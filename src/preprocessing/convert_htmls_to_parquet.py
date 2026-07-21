from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from pathlib import Path
from pyarrow.parquet import ParquetWriter
import pyarrow as pa
import math
import re 
from selectolax.lexbor import LexborHTMLParser

from scripts.data_generation_config import (
    BATCH_SIZE, 
    N_WORKERS_FILE_LOADING, 
    HTML_CACHE_DIR, 
    HTML_PARQUET_PATH
)

CHARS_TO_REMOVE = re.compile("[\n|\t|\xa0| ]+")

def clean_html(html: str) -> str: 
    text = LexborHTMLParser(html)
    text.strip_tags(['style', 'script'])

    text = CHARS_TO_REMOVE.sub(" ", text)

    return text

def load_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def preprocess_file(path: Path) -> str: 
    return clean_html(
        load_file(path)
    )

def batch(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk

def convert_htmls_to_parquet(): 
    files = list(Path(HTML_CACHE_DIR).glob("*.html"))
    total_files = math.ceil(len(files) / BATCH_SIZE)
    schema = pa.schema([('hash', pa.string()), ('html_data', pa.large_string())])
    i = 1

    with ParquetWriter(HTML_PARQUET_PATH, schema, compression="zstd") as writer:
        with ProcessPoolExecutor(max_workers=N_WORKERS_FILE_LOADING) as pool: 
            for file_batch in batch(files, BATCH_SIZE): 
                print(f"Processing batch {i}/{total_files}")
                results = list(pool.map(preprocess_file, file_batch, chunksize=64))
                file_names = list(map(lambda x: x.stem, file_batch))
                table = pa.Table.from_arrays([file_names, results], schema=schema)
                writer.write_table(table, row_group_size=512)
                i+= 1

if __name__ == '__main__': 
    convert_htmls_to_parquet()

