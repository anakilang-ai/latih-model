import pandas as pd
import numpy as np
import logging
import csv
import torch
import evaluate
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, GenerationConfig
from sklearn.model_selection import train_test_split as tts
from utils import QADataset, logging_config

# Logging configuration
logging_config('log_model', 'training.log')

# Function to filter valid rows
def filter_valid_rows(row):
    return len(row) == 2 and all(row)

def prepare_data(file_path, delimiter='|', encoding='utf-8'):
    filtered_rows = []
    with open(file_path, 'r', encoding=encoding) as file:
        reader = csv.reader(file, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        for row in reader:
            if filter_valid_rows(row):
                filtered_rows.append(row)

    df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
    return df

def train_model(df, model_name='facebook/bart-base', epoch=20, batch_size=10):
    # Split dataset into training and test sets
    train_df, test_df = tts(df, test_size=0.2, random_state=42)

    # Reset index to ensure continuous indexing
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Prepare the dataset
    tokenizer = BartTokenizer.from_pretrained(model_name)
    inputs_train = train_df['question'].tolist()
    targets_train = train_df['answer'].tolist()
    inputs_test = test_df['question'].tolist()
    targets_test = test_df['answer'].tolist()

    dataset_train = QADataset(inputs_train, targets_train, tokenizer, max_length=160)
    dataset_test = QADataset(inputs_test, targets_test, tokenizer, max_length=160)

    # Load model
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Define data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./result/results_coba-{epoch}-{batch_size}',
        num_train_epochs=epoch,
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

    # Define generation config
    generation_config = GenerationConfig(
        early_stopping=True,
        num_beams=5,
        no_repeat_ngram_size=0,
        forced_bos_token_id=0,
        forced_eos_token_id=2,
        max_length=160,
        bos_token_id=0,
        decoder_start_token_id=2
    )

    # Load metrics
    bleu_metric = evaluate.load("bleu")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]

        # Convert logits to a tensor
        logits = torch.tensor(logits)
        predictions = torch.argmax(logits, dim=-1)
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # BLEU score
        bleu = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

        return {
            "bleu": bleu["bleu"],
        }

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the model
model.save_pretrained(path)
tokenizer.save_pretrained(path)
generation_config.save_pretrained(path)

    # Evaluate model
    eval_results = trainer.evaluate()

    # Print evaluation results
print("Evaluation Results:")
for metric, score in eval_results.items():
    print(f"{metric}: {score:.4f}")

# Log the model path and evaluation results
logging.info("Model Path: %s", path)
logging.info("Evaluation Results:")
for metric, score in eval_results.items():
    logging.info(f"{metric}: {score:.4f}")
logging.info("------------------------------------------\n")


if __name__ == "__main__":
    # Load dataset and train model
    dataset_file_path = 'dataset-kelas.csv'
    df = prepare_data(dataset_file_path)
    train_model(df)
