from scripts.data_generation_config import TRAINING_READY_DATASET_PATH
from training_config import (
    MODEL_NAME, 
    CHECKPOINT_DIR, 
    SAVE_STEPS, 
    MAX_TRAINING_STEPS, 
    LOGGING_STEPS
)
from dataloader import tokenizer, process_batch
from datasets import load_dataset
from transformers import (
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq, 
    AutoModelForCausalLM
)

def train(): 
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    dataset = load_dataset("parquet", 
                           data_files=TRAINING_READY_DATASET_PATH, 
                           split="train", 
                           streaming=True)

    dataset = dataset.shuffle(seed=42, buffer_size=10_000)
    dataset = dataset.map(process_batch, batched=True)

    collator = DataCollatorForSeq2Seq(tokenizer, 
                                      padding=True, 
                                      label_pad_token_id=-100)

    training_arguments = TrainingArguments(
        output_dir=CHECKPOINT_DIR, 
        save_steps=SAVE_STEPS, 
        max_steps=MAX_TRAINING_STEPS, 
        logging_steps=LOGGING_STEPS
    )

    trainer = Trainer(
        model=model, 
        args=training_arguments, 
        data_collator=collator, 
        train_dataset=dataset
    )

    trainer.train()

if __name__ == '__main__': 
    train()