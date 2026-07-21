from scrape_htmls import scrape_htmls
from convert_htmls_to_parquet import convert_htmls_to_parquet
from prepare_dataset import prepare_dataset
from merge_htmls_with_dataset import merge_htmls_with_dataset

prepare_dataset()
scrape_htmls()
# convert_htmls_to_parquet()
# merge_htmls_with_dataset()
