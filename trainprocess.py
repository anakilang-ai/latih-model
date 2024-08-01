import pandas as pd
import numpy as np
import logging
import csv
import torch
import evaluate
import os
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig
)
from sklearn.model_selection import train_test_split
from utils import QADataset, logging_config

# Logging configuration
logging_config('log_model', 'training.log')

def filter_valid_rows(row):
    """Ensure each row has exactly two non-empty elements."""
    return len(row) == 2 and all(row)

def load_and_filter_dataset(file_path):
    """Load and filter dataset from CSV file."""
    filtered_rows = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|', quoting=csv.QUOTE_NONE)
        filtered_rows = [row for row in reader if filter_valid_rows(row)]
    return pd.DataFrame(filtered_rows, columns=['question', 'answer'])

def split_dataset(df, test_size=0.2, random_state=42):
    """Split the dataset into training and test sets."""
    return train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)

def prepare_datasets(train_df, test_df, tokenizer, max_length=160):
    """Prepare training and test datasets."""
    inputs_train = train_df['question'].tolist()
    targets_train = train_df['answer'].tolist()
    inputs_test = test_df['question'].tolist()
    targets_test = test_df['answer'].tolist()

    dataset_train = QADataset(inputs_train, targets_train, tokenizer, max_length=max_length)
    dataset_test = QADataset(inputs_test, targets_test, tokenizer, max_length=max_length)
    
    return dataset_train, dataset_test

def configure_training_args(output_dir, num_train_epochs, batch_size):
    """Setup training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=160,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=160,
        save_total_limit=2,
        fp16=True,
        evaluation_strategy="epoch",
    )

def setup_generation_config():
    """Define and return generation configuration."""
    return GenerationConfig(
        early_stopping=True,
        num_beams=5,
        no_repeat_ngram_size=0,
        forced_bos_token_id=0,
        forced_eos_token_id=2,
        max_length=160,
        bos_token_id=0,
        decoder_start_token_id=2
    )

def compute_bleu_metrics(eval_pred, tokenizer):
    """Compute BLEU score for evaluation."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = torch.tensor(logits)
    predictions = torch.argmax(logits, dim=-1)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu_metric = evaluate.load("bleu")
    bleu_score = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

    return {
        "bleu": bleu_score["bleu"],
    }

def main():
    num = 'dataset-kelas'
    model_name = 'facebook/bart-base'
    output_dir = f'./result/results_coba{num}-{epoch}-{batch_size}'

    # Load and prepare dataset
    df = load_and_filter_dataset(f'{num}.csv')
    train_df, test_df = split_dataset(df)

    tokenizer = BartTokenizer.from_pretrained(model_name)
    dataset_train, dataset_test = prepare_datasets(train_df, test_df, tokenizer)

    # Initialize model
    model = BartForConditionalGeneration.from_pretrained(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Training arguments
    epoch = 20
    batch_size = 10
    training_args = configure_training_args(output_dir, num_train_epochs=epoch, batch_size=batch_size)

    # Generation configuration
    generation_config = setup_generation_config()

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_bleu_metrics(eval_pred, tokenizer)
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model_save_path = f'model/bart_coba{num}-{epoch}-{batch_size}'
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    generation_config.save_pretrained(model_save_path)

    # Evaluate model
    eval_results = trainer.evaluate()

    # Print and log evaluation results
    print(f"Evaluation results: {eval_results}")
    logging.info(f"Model saved to: {model_save_path}")
    logging.info(f"Evaluation results: {eval_results}")
    logging.info("------------------------------------------\n")

if __name__ == "__main__":
    main()
