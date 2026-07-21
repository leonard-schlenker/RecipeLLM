HTML_CACHE_DIR = "/run/media/lenni/HDD1GB/html_files"
CLEANED_DATASET_PATH = "./data/dataset/RecipeNLG_dataset_cleaned.parquet"
ORIG_DATASET_PATH = "./data/raw/RecipeNLG_dataset.csv"

# downloading html files containing the recipe in humand-readable format
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
FAILED_LOG = "/run/media/lenni/HDD1GB/failed_downloads.log"
FILE_LIMIT = 3_000_000  # target total of html files in HTML_CACHE_DIR, counting already-cached ones; None = download everything
CONCURRENT_REQUESTS = 200
CONCURRENT_PER_DOMAIN = 16
PROGRESS_EVERY = 100
SEED_YIELD_EVERY = 200
RETRY_BATCH_SIZE = 10_000  # failed URLs re-scheduled per idle event in the retry phase
TENSORBOARD_LOG_DIR = "./runs/scrape"  # each run logs to a timestamped subdir; view with: tensorboard --logdir runs/scrape
TENSORBOARD_INTERVAL = 60  # seconds between scalar samples; 60 keeps the *_per_min tags honest

# converting raw html to parquet dataset
N_WORKERS_FILE_LOADING = 10 # number of worker threads that load the html batch to memory
BATCH_SIZE = 10_000 # number of files to load simultaneously into memory to build a partial table from 
HTML_PARQUET_PATH = "/run/media/lenni/HDD1GB/htmls.parquet" # file to write the htmls table to 

TRAINING_READY_DATASET_PATH = "./data/dataset/RecipeNLG_training_dataset.parquet"
