HTML_CACHE_DIR = "/run/media/lenni/HDD1GB/temp_html_files/"
CLEANED_DATASET_PATH = "./data/raw/RecipeNLG_dataset.parquet"
ORIG_DATASET_PATH = "./data/raw/RecipeNLG_dataset.csv"

# downloading html files containing the recipe in humand-readable format
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
FAILED_LOG = "./data/raw/failed_downloads.log"
CONCURRENT_REQUESTS = 200
CONCURRENT_PER_DOMAIN = 16
PROGRESS_EVERY = 100
SEED_YIELD_EVERY = 200

# converting raw html to parquet dataset
N_WORKERS_FILE_LOADING = 25 # number of worker threads that load the html batch to memory
BATCH_SIZE = 4000 # number of files to load simultaneously into memory to build a partial table from 
HTML_PARQUET_PATH = "./data/dataset/small_htmls.parquet" # file to write the htmls table to 

TRAINING_READY_DATASET_PATH = "./data/dataset/RecipeNLG_training_dataset.parquet"
