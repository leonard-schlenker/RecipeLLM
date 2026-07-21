"""
The goal of this script is to prepare the downloaded RecipeNLG dataset for scraping and dataset creation. 
"""

import polars as pl
from scripts.data_generation_config import ORIG_DATASET_PATH, CLEANED_DATASET_PATH
from typing import Tuple

def format_links(df: pl.LazyFrame) -> pl.LazyFrame: 
    df = df.with_columns(
        pl.when(
            pl.col("link").str.starts_with("www.")
        ).then(
            pl.col("link")
        ).otherwise(
            pl.concat_str(pl.lit("www."), pl.col("link"))
        )
    )

    return df

def extract_websites(df: pl.LazyFrame) -> pl.LazyFrame: 
    df = df.with_columns(
        pl.col("link").str.extract(r"www\..*\.[a-z]{2,3}/", 0).alias("website")
    )

    return df

def build_responses(df: pl.LazyFrame) -> pl.LazyFrame: 
    df = df.with_columns(
        pl.concat_str(
            pl.col("title"), 
            pl.lit("\n"), 
            pl.col("ingredients"), 
            pl.lit("\n"), 
            pl.col("directions")
        ).alias("response")
    )

    return df

def deduplicate(df: pl.LazyFrame) -> pl.LazyFrame: 

    df = _sort_df_by_website_distr(df)

    df = _remove_exact_duplicates(df)

    ...

def _sort_df_by_website_distr(df: pl.LazyFrame, descending=False) -> pl.LazyFrame: 
    website_distr = df.group_by("website").len()

    df = df.join(website_distr, on="website", how="inner")

    df = df.sort(by="len")

    df = df.drop("len")

    return df 

def _remove_exact_duplicates(df: pl.LazyFrame) -> pl.LazyFrame: 
    # we keep the first because we previously sorted the rows by how many websites their website of origin provides

    # by keeping the first we choose recipes from websites that deliver few samples thereby moving towards more 
    # uniformyly distributed dataset and keeping extracted structure of recipes more diverse 
    return df.unique(subset=pl.col("response"), keep="first")
    
def compute_link_hashes(df: pl.LazyFrame) -> pl.LazyFrame:
    # Native polars hash runs parallel over the column, unlike a per-row
    # hashlib map_elements. Stored as string because downstream the hash is
    # both a cache filename stem and a join key. Polars hashes are not stable
    # across polars versions, so this stored column is the single source of
    # truth — never recompute it, always read it back (scrape_htmls.py does).
    df = df.with_columns(
        pl.col("link").hash(seed=0).cast(pl.String).alias("link_hash")
    )

    return df

def compute_dataset_split(df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]: 

    # we take ~10% of the dataset for evaluation 
    # this is about the first 16 websites due to the sample distribution 

    website_groups = df.group_by("website").len()

    website_groups = website_groups.with_columns(
        (pl.col("len") / len(df)).alias("frequency")
    )

    eval_websites = website_groups.sort("frequency").head(16)["website"]

    df = df.with_columns(
        pl.col("website").is_in(eval_websites).alias("eval")
    )

    training, eval = df.partition_by("eval", as_dict=True, include_key=False).values()

    return training, eval 

def sample_by_website(df, n): 
    df = df.partition_by("website")
    partitions = [p.sample(n) for p in df]
    df = pl.concat(partitions, how='vertical')
    return df 

def get_website_distrbution(df): 
    return df.group_by("website").len()

def prepare_dataset(): 
    df = pl.read_csv(ORIG_DATASET_PATH)

    df = df.lazy()

    df = format_links(df)
    df = extract_websites(df)

    df = build_responses(df)
    df = compute_link_hashes(df)

    df = df.select(
        pl.col("link"),
        pl.col("website"), 
        pl.col("link_hash"), 
        pl.col("response")
    )

    df = df.collect()

    training, eval = compute_dataset_split(df)

    training_websites = get_website_distrbution(training)
    training_min_sample_size = training_websites["len"].min()

    eval_websites = get_website_distrbution(eval)
    eval_min_sample_size = eval_websites["len"].min()

    training = sample_by_website(training, training_min_sample_size)
    eval = sample_by_website(eval, eval_min_sample_size)

    training = training.with_columns(
        pl.lit(True).alias("train")
    )

    eval = eval.with_columns(
        pl.lit(False).alias("train")
    )

    training = training.vstack(eval)

    training.write_parquet(CLEANED_DATASET_PATH)


if __name__ == '__main__': 
    prepare_dataset()

