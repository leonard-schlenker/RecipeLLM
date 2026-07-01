import polars as pl
from scripts.prep_dataframe import prepare_df, load_htmls, build_targets, build_data_tensors
from dataloader import prepare_dataloading, df_to_parquet
from transformers import AutoTokenizer

def main(): 
    df = pl.read_csv("./data/dataset/RecipeNLG_dataset_prep.csv")

    tok, prep = prepare_dataloading(df)

    df_to_parquet(df, prep, "./data/dataset/recipes.parquet")
    ...

if __name__ == '__main__': 
    main()