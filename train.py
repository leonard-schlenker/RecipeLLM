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
from peft import LoraConfig, PeftModel, get_peft_model

def train(): 

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    model = get_peft_model(model, lora_config)

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