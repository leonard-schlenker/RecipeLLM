"""
The goal of this script is to prepare the downloaded RecipeNLG dataset for scraping and dataset creation. 
"""

import polars as pl 
from data_generation_config import CLEANED_DATASET_PATH, ORIG_DATASET_PATH
import hashlib

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

def compute_link_hashes(df: pl.LazyFrame) -> pl.LazyFrame: 
    df = df.with_columns(
        pl.concat_str(
            pl.lit("http://"), 
            pl.col("link")
        ).map_elements(
            lambda url: hashlib.md5(url.encode()).hexdigest()
        ).alias("hash")
    )

    return df 

def prepare_dataset(): 
    df = pl.read_csv(ORIG_DATASET_PATH)

    df = df.lazy()

    df = format_links(df)
    df = extract_websites(df)
    df = build_responses(df)
    df = compute_link_hashes(df)

    df = df.select(
        pl.col("hash"), 
        pl.col("response")
    )

    df = df.collect()

    df.write_parquet(CLEANED_DATASET_PATH)


if __name__ == '__main__': 
    prepare_dataset()

