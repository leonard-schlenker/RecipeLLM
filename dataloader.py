from transformers import AutoTokenizer
from training_config import MODEL_NAME
import polars as pl

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def process_batch(batch): 
    message_dtype = pl.Struct([pl.Field("content", pl.String), pl.Field("role", pl.String)])
    batch["input"] = pl.DataFrame(pl.Series(values=batch["input"], name="htmls")).with_columns(
        pl.concat_list(
            pl.col("htmls").str.json_decode(message_dtype)
        ).alias("htmls")
    )["htmls"].to_list()

    input_ids = tokenizer.apply_chat_template(batch["input"], 
                                              tokenize=False, 
                                              add_generation_prompt=True)

    input_ids = tokenizer(input_ids, 
                          return_attention_mask=False)["input_ids"]

    response_ids = tokenizer([r + tokenizer.eos_token for r in batch["response"]], # batch["response"] + [tokenizer.eos_token], 
                             return_attention_mask=False)["input_ids"]

    input_ids = [i + r for i, r in zip(input_ids, response_ids)]

    labels = [[-100] * len(i) + r for i, r in zip(input_ids, response_ids)]

    return {"input_ids": input_ids, "labels": labels}

