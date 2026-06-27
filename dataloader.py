import polars as pl
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from datasets import IterableDataset, Dataset

def df_to_parquet(df: pl.DataFrame, prep: callable, parquet_path: str, n_proc: int=1, batched: bool=True):
    ds = IterableDataset.from_polars(df)

    ds = ds.map(prep, batched=batched, remove_columns=["input", "target"])

    # `datasets.to_parquet` reserves `compression` internally, so write with pyarrow
    # directly. zstd at a high level gives near-best ratio with fast decompression.
    table = Dataset.from_list(list(ds)).data.table
    pq.write_table(table, parquet_path, compression="zstd", compression_level=19)

def prepare_dataloading(df: pl.DataFrame): 

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    def preprocess(batch): 

        prompt = batch["input"]

        response = batch["target"]

        input_ids = tokenizer(prompt, 
                              add_special_tokens=False, 
                              return_attention_mask=False)["input_ids"]

        target_ids = tokenizer(response, 
                               add_special_tokens=False, 
                               return_attention_mask=False)["input_ids"]

        total_ids = [ipt + tgt for ipt, tgt in zip(input_ids, target_ids)]

        labels = total_ids.copy()

        for i in range(len(labels)): 
            labels[i][:len(input_ids)] = [-100] * len(input_ids)

        return {"input_ids": total_ids, "labels": labels}

    return tokenizer, preprocess
    