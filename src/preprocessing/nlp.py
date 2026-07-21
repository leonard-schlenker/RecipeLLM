from spacy import load as spacy_load
from spacy.matcher import DependencyMatcher
import polars as pl
from typing import List, Tuple

SPACY_MODEL = "en_core_web_md"

BATCH_SIZE = 256

SEARCH_PATTERN = [
    {"RIGHT_ID": "verb", "RIGHT_ATTRS": {"POS": "ROOT", "DEP": "VERB"}}, 
    {"LEFT_ID": "verb", "REL_OP": ">", "RIGHT_ID": "prt", "RIGHT_ATTRS": {"POS": "prt"}}
]

def get_verb_prt_combinations(in_df: pl.DataFrame) -> pl.DataFrame: 
    in_df = in_df.with_row_index(name="idx")

    df = _format_data_frame(in_df)
    texts = df["directions"].to_list()
    gen, matcher = _build_doc_generator_and_matcher(texts)

    for direction_id, doc in enumerate(gen): 
        for _, token_ids in matcher(doc): 
            df["directions"][direction_id] = [doc[idx] for idx in token_ids]

    df.group_by(pl.col("idx")).agg(pl.col("directions"))

    in_df = in_df.drop("directions")

    in_df = in_df.join(df, on="idx", how="inner")
            
    return in_df

def _format_data_frame(df: pl.DataFrame) -> pl.DataFrame: 

    df = df.lazy()

    df = df.select(
            [pl.col("idx"), pl.col("directions")]
        ).with_columns(
            pl.col("directions").str.json_decode(pl.List(pl.String))
        ).explode(
            pl.col("directions")
        ).with_columns(
            pl.col(
                "directions"
            ).str.replace_all(
                r"; ", 
                ". "
            ).str.split(
                by=".", 
                inclusive=True
            )
        ).explode(
            pl.col("directions")
        ).with_columns(
            pl.col("directions").str.strip_chars()
        )

    return df.collect()

def _build_doc_generator_and_matcher(texts: List[str]): 
    nlp = spacy_load(SPACY_MODEL)
    docs_generator = nlp.pipe(
        texts, 
        batch_size=512, 
        n_process=4
    )

    matcher = DependencyMatcher(nlp.vocab)
    matcher.add("VERB_PRT_PAT", [SEARCH_PATTERN])

    return docs_generator, matcher

if __name__ == '__main__': 
    from data_generation_config import ORIG_DATASET_PATH
    df = pl.read_csv(ORIG_DATASET_PATH, n_rows=10)

    df = get_verb_prt_combinations(df)

    print(df.select(pl.col("directions")))