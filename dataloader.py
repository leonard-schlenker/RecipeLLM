from transformers import AutoTokenizer
from training_config import MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def process_batch(batch): 

    input_ids = tokenizer.apply_chat_template(batch["input"], 
                                              tokenize=False, 
                                              add_generation_prompt=True, 
                                              )

    input_ids = tokenizer(input_ids, 
                          return_attention_mask=False)["input_ids"]

    response_ids = tokenizer(batch["response"] + [tokenizer.eos_token], 
                             return_attention_mask=False)["input_ids"]

    labels = [-100] * len(input_ids) + response_ids 

    input_ids += response_ids

    return {"input_ids": input_ids, "labels": labels}

