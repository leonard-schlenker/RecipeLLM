import polars as pl
from data_generation_config import (
    CLEANED_DATASET_PATH, 
    HTML_PARQUET_PATH, 
    TRAINING_READY_DATASET_PATH
) 

def merge_htmls_with_dataset(): 
    dataset = pl.scan_parquet(CLEANED_DATASET_PATH)
    htmls = pl.scan_parquet(HTML_PARQUET_PATH, low_memory=True)

    # only leave the hash so that we can join and reduce `dataset` to 
    # only the recipes that we already scraped
    htmls = htmls.select(
        pl.col("hash")
    )

    """
    dataset = htmls.join(dataset, 
                   on="hash", 
                   how="inner").select(
                       pl.col("response"), 
                       pl.struct(role=pl.lit("user"),
                                 content=pl.col("html_data")).alias("input")# .struct.json_encode().alias("input")
                ).sink_parquet(TRAINING_READY_DATASET_PATH, engine="streaming", compression="zstd")
    """

    dataset = htmls.join(
        dataset, 
        on='hash', 
        how='inner'
    )

    # find the smallest group
    domain_freqs = dataset.group_by(
        pl.col("website")
    ).len()

    min_len = domain_freqs.select(pl.col("len")).min()

    # sample uniformly from each website using `min_len`
    website_partitions = dataset.collect().partition_by(
        by="website", 
        include_key=True
    )

    website_partitions = [web_df.sample(min_len) for web_df in website_partitions]

    dataset = pl.concat(website_partitions, how='vertical')

    htmls = pl.scan_parquet(HTML_PARQUET_PATH)

    dataset = htmls.join(dataset, 
                         on='hash', 
                         how='inner'
                        ).select(
                             pl.col("response"), 
                             pl.struct(role=pl.lit("user"), 
                                       content=pl.col("html_data")).alias("input")
                        ).sink_parquet(TRAINING_READY_DATASET_PATH, 
                                       engine='streaming', 
                                       compression='zstd'
                        )

if __name__ == '__main__': 
    merge_htmls_with_dataset()
