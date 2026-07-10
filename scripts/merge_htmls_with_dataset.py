import polars as pl
from data_generation_config import (
    CLEANED_DATASET_PATH, 
    HTML_PARQUET_PATH, 
    TRAINING_READY_DATASET_PATH
) 

def merge_htmls_with_dataset(): 
    dataset = pl.scan_parquet(CLEANED_DATASET_PATH)
    htmls = pl.scan_parquet(HTML_PARQUET_PATH, low_memory=True)

    dataset = htmls.join(dataset, 
                   on="hash", 
                   how="inner").select(
                       pl.col("response"), 
                       pl.struct(role=pl.lit("user"),
                                 content=pl.col("html_data")).alias("input")# .struct.json_encode().alias("input")
                ).sink_parquet(TRAINING_READY_DATASET_PATH, engine="streaming", compression="zstd")


if __name__ == '__main__': 
    merge_htmls_with_dataset()