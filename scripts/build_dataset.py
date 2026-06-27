import polars as pl
from transformers import AutoTokenizer
import json
import urllib.request
import os
import hashlib
import torch 
from typing import Tuple, List

HTML_CACHE_DIR = "./data/raw/html_cache"

def fetch_or_load(url: str) -> str:
    os.makedirs(HTML_CACHE_DIR, exist_ok=True)
    filename = hashlib.md5(url.encode()).hexdigest() + ".html"
    cache_path = os.path.join(HTML_CACHE_DIR, filename)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    html = urllib.request.urlopen(url).read().decode("utf-8")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html

def prepare_df(df): 

    df = df.with_columns(
        pl.when(pl.col("link").str.starts_with("www.")).then(
            pl.col("link")
        ).otherwise(
            pl.lit("www.") + pl.col("link")
        )
    )

    df = df.with_columns(
        pl.col("link").str.extract(r"www\..*\.[a-z]{2,3}/", 0).alias("website")
    )

    return df 

def build_targets(df): 

    df = df.with_columns(
        pl.concat_str(
            pl.col("title"), 
            pl.lit("\n"), 
            pl.col("ingredients"), 
            pl.lit("\n"), 
            pl.col("directions")
        ).alias("target")
    )

    return df 

def load_htmls(df) -> List[str]: 
    htmls = []
    for link in df["link"]:
        htmls.append(fetch_or_load(f"http://{link}"))

    df = df.with_columns(
        pl.Series(name="input", values=htmls)
    )

    return df 

def build_data_tensors(df: pl.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]: 
    assert "input" in df.columns
    assert "target" in df.columns

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    df = df.with_columns(
        pl.struct(
            role=pl.lit("user"), 
            content=pl.col("input")
        ).alias("message")
    )

    inputs = df["message"].to_list()

    inputs = tokenizer(inputs, 
                       tokenize=False, 
                       add_generation_prompt=True)

    df = df.with_columns(
        pl.Series(name="message", values=inputs)
    )

    df = df.with_columns(
        pl.str.concat(pl.col("message"), pl.col("target"), pl.lit(tokenizer.eos_token))
    )

    return df 
        

def main(): 
    df = pl.read_csv("./data/raw/RecipeNLG_dataset.csv", n_rows=1000)
    # df = pl.read_csv("./data/RecipeNLG_dataset.csv")

    df = build_targets(df)

    df = load_htmls(df)

    df = df.with_columns(
        pl.struct(
            role=pl.lit("user"), 
            content=pl.col("input")
        ).alias("message")
    )

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    df = df.with_columns(
        pl.concat("message")
    )

    inputs = df["message"].to_list()

    inputs = tok.apply_chat_template(
        inputs, 
        tokenize=False, 
        add_generation_prompt=True
    )

    df = df.with_columns(
        pl.Series(name="input", values=inputs), 
        pl.concat_str(pl.col("target"), pl.lit(tok.eos_token))
    )

    df = df.select(
        pl.col("input"), 
        pl.col("target")
    )

    df.write_csv("./data/dataset/RecipeNLG_dataset_prep.csv")



