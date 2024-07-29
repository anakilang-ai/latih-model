import pandas as pd
import numpy as np
import logging
import csv
import torch
import evaluate
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, GenerationConfig
from sklearn.model_selection import train_test_split as tts
from utils import QADataset, logging_config

# Set up logging
logging_config('log_model', 'training.log')

# Function to validate rows
def is_valid_row(row):
    return len(row) == 2 and all(row)

# Load dataset
file_prefix = 'dataset-kelas'
valid_rows = []
with open(f'{file_prefix}.csv', 'r', encoding='utf-8') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_NONE)
    for row in csv_reader:
        if is_valid_row(row):
            valid_rows.append(row)

data = pd.DataFrame(valid_rows, columns=['question', 'answer'])

# Split data into training and testing sets
train_data, test_data = tts(data, test_size=0.2, random_state=42)

# Reset index for consistent indexing
train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Prepare tokenizer and dataset
model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)

train_questions = train_data['question'].tolist()
train_answers = train_data['answer'].tolist()

test_questions = test_data['question'].tolist()
test_answers = test_data['answer'].tolist()

train_dataset = QADataset(train_questions, train_answers, tokenizer, max_length=160)
test_dataset = QADataset(test_questions, test_answers, tokenizer, max_length=160)

# Load the BART model
model = BartForConditionalGeneration.from_pretrained(model_name)

# Data collator setup
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# Training parameters
num_epochs = 20
train_batch_size = 10

# Training configuration
training_params = TrainingArguments(
    output_dir=f'./result/results_coba{file_prefix}-{num_epochs}-{train_batch_size}',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=train_batch_size,
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

# Generation configuration
gen_config = GenerationConfig(
    early_stopping=True,
    num_beams=5,
    no_repeat_ngram_size=0,
    forced_bos_token_id=0,
    forced_eos_token_id=2,
    max_length=160,
    bos_token_id=0,
    decoder_start_token_id=2
)

# Load evaluation metric
bleu = evaluate.load("bleu")

def calculate_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    # Convert logits to tensor
    logits = torch.tensor(logits)
    preds = torch.argmax(logits, dim=-1)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute BLEU score
    bleu_score = bleu.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])

    return {
        "bleu": bleu_score["bleu"],
    }

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_params,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator,
    compute_metrics=calculate_metrics
)

# Start training
trainer.train()

# Save model and tokenizer
model_dir = f'model/bart_coba{file_prefix}-{num_epochs}-{train_batch_size}'
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
gen_config.save_pretrained(model_dir)

# Evaluate the model
evaluation_results = trainer.evaluate()

# Log evaluation results
print(f"Evaluation results: {evaluation_results}")
logging.info(f"Model: {model_dir}")
logging.info(f"Evaluation results: {evaluation_results}")
logging.info("------------------------------------------\n")