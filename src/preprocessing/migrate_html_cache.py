"""One-off migration: rename scraped HTML files from the legacy md5-based
filenames to the polars-hash filenames the current pipeline uses.

Legacy scheme (old scrape_htmls.py, see commit 18c9070):

    md5(f"http://{link}".encode()).hexdigest() + ".html"

Current scheme (prepare_dataset.py):

    str(link.hash(seed=0)) + ".html"

with ``link`` www.-normalised in both schemes.

The old->new mapping is built from the *full* source dataset
(ORIG_DATASET_PATH), not just the cleaned parquet: the per-website sampling in
prepare_dataset.py is unseeded, so links scraped under an older parquet may no
longer be in the current one — building from the full dataset renames those
files too instead of leaving them behind.

That requires recomputing the polars hash, which is only safe if this polars
version reproduces the stored hashes bit-for-bit (polars hashes are not stable
across versions). So before touching anything the recomputed hashes are
validated against the ``hash`` column stored in CLEANED_DATASET_PATH for every
link in it; any mismatch aborts the run.

Safety properties:
  - dry-run by default; pass --apply to actually rename
  - aborts if two distinct on-disk files would rename to the same target
  - never overwrites: if the target name already exists, the file is skipped
    and reported as a conflict
  - os.rename within the same directory: atomic, no data copied
  - files matching neither scheme are left untouched and reported

Usage:

    python scripts/migrate_html_cache.py --dir data/raw/html_cache          # dry run
    python scripts/migrate_html_cache.py --dir data/raw/html_cache --apply
"""

import argparse
import hashlib
import os

import polars as pl

from data_generation_config import (
    HTML_CACHE_DIR,
    CLEANED_DATASET_PATH,
    ORIG_DATASET_PATH,
)


def load_all_links_with_hashes() -> pl.DataFrame:
    """(link, hash) for every link in the full source CSV, www.-normalised and
    hashed exactly as prepare_dataset.py does."""
    return (
        pl.scan_csv(ORIG_DATASET_PATH)
        .select(
            pl.when(pl.col("link").str.starts_with("www."))
            .then(pl.col("link"))
            .otherwise(pl.concat_str(pl.lit("www."), pl.col("link")))
            .alias("link")
        )
        .unique(subset="link")
        .with_columns(pl.col("link").hash(seed=0).cast(pl.String).alias("hash"))
        .collect()
    )


def validate_against_stored_hashes(recomputed: pl.DataFrame) -> None:
    """Abort unless the recomputed hashes reproduce the stored `hash` column
    for every link in the cleaned parquet. This is the guard that makes
    recomputation trustworthy: if it fails, this polars version hashes
    differently and the new filenames would not match what the pipeline
    (pending_downloads etc.) reads back from the parquet."""
    stored = pl.read_parquet(CLEANED_DATASET_PATH, columns=["link", "hash"]).unique(
        subset="link"
    )
    joined = stored.join(recomputed, on="link", how="left", suffix="_recomputed")
    missing = joined.filter(pl.col("hash_recomputed").is_null())
    if len(missing) > 0:
        raise SystemExit(
            f"ABORT: {len(missing)} links from {CLEANED_DATASET_PATH} not found in "
            f"{ORIG_DATASET_PATH} (first: {missing['link'][0]!r}). "
            "The datasets are out of sync; refusing to rename anything."
        )
    mismatched = joined.filter(pl.col("hash") != pl.col("hash_recomputed"))
    if len(mismatched) > 0:
        row = mismatched.row(0, named=True)
        raise SystemExit(
            f"ABORT: recomputed polars hash differs from the stored hash column for "
            f"{len(mismatched)} of {len(joined)} links "
            f"(e.g. {row['link']!r}: stored {row['hash']} != recomputed {row['hash_recomputed']}). "
            "Your polars version does not reproduce the stored hashes; refusing to rename anything."
        )
    print(
        f"Hash validation OK: recomputed hashes match the stored `hash` column "
        f"for all {len(joined):,} links in {CLEANED_DATASET_PATH}."
    )


def build_rename_map(links_with_hashes: pl.DataFrame) -> dict[str, str]:
    """old md5 filename -> new polars-hash filename, for every known link."""
    mapping: dict[str, str] = {}
    for link, link_hash in links_with_hashes.rows():
        old = hashlib.md5(f"http://{link}".encode()).hexdigest() + ".html"
        new = link_hash + ".html"
        existing = mapping.get(old)
        if existing is not None and existing != new:
            raise SystemExit(
                f"ABORT: md5 collision — {old} maps to both {existing} and {new}."
            )
        mapping[old] = new
    return mapping


def migrate(directory: str, apply: bool) -> None:
    links_with_hashes = load_all_links_with_hashes()
    validate_against_stored_hashes(links_with_hashes)
    mapping = build_rename_map(links_with_hashes)
    new_names = set(mapping.values())

    files = set(os.listdir(directory))
    to_rename: list[tuple[str, str]] = []
    conflicts: list[tuple[str, str]] = []
    already_migrated = 0
    for name in files:
        if name in new_names:
            already_migrated += 1
            continue
        new = mapping.get(name)
        if new is None:
            continue  # counted as unmatched below
        if new in files:
            conflicts.append((name, new))
        else:
            to_rename.append((name, new))
    unmatched = len(files) - already_migrated - len(to_rename) - len(conflicts)

    # Two distinct old files must never race for the same target name — the
    # second os.rename would silently replace the first one's data.
    targets = [new for _, new in to_rename]
    if len(set(targets)) != len(targets):
        raise SystemExit(
            "ABORT: two distinct files map to the same target name; renaming "
            "would overwrite data."
        )

    print(
        f"\n{directory}: {len(files):,} files total\n"
        f"  {len(to_rename):,} legacy md5 names to rename\n"
        f"  {already_migrated:,} already using the current hash names\n"
        f"  {len(conflicts):,} conflicts (target name already exists — skipped)\n"
        f"  {unmatched:,} matching neither scheme (left untouched)"
    )
    for old, new in conflicts[:10]:
        print(f"  CONFLICT: {old} -> {new} (target exists)")
    for old, new in to_rename[:3]:
        print(f"  example: {old} -> {new}")

    if not apply:
        print("\nDry run — nothing renamed. Re-run with --apply to rename.")
        return

    for old, new in to_rename:
        os.rename(os.path.join(directory, old), os.path.join(directory, new))
    print(f"\nRenamed {len(to_rename):,} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir",
        default=HTML_CACHE_DIR,
        help=f"directory holding the scraped .html files (default: {HTML_CACHE_DIR})",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually rename files (default is a dry run)",
    )
    args = parser.parse_args()
    migrate(args.dir, args.apply)
