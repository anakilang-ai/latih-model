import pandas as pd
import numpy as np
import logging
import csv
import torch
import evaluate
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, GenerationConfig
from sklearn.model_selection import train_test_split
from utils import QADataset, setup_logging

# Logging setup
setup_logging('log_model', 'training.log')

# Function to validate rows
def is_valid_row(row):
    return len(row) == 2 and all(row)

# Load dataset
dataset_name = 'dataset-kelas'
valid_rows = []
with open(f'{dataset_name}.csv', 'r', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile, delimiter='|', quoting=csv.QUOTE_NONE)
    for row in csvreader:
        if is_valid_row(row):
            valid_rows.append(row)

data = pd.DataFrame(valid_rows, columns=['question', 'answer'])

# Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Reset index
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Prepare the tokenizer and datasets
model_identifier = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_identifier)

train_inputs = train_data['question'].tolist()
train_targets = train_data['answer'].tolist()
test_inputs = test_data['question'].tolist()
test_targets = test_data['answer'].tolist()

train_dataset = QADataset(train_inputs, train_targets, tokenizer, max_length=160)
test_dataset = QADataset(test_inputs, test_targets, tokenizer, max_length=160)

# Load the model
model = BartForConditionalGeneration.from_pretrained(model_identifier)

# Data collator
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training parameters
epochs = 20
batch_size_train = 10

# Training arguments
train_args = TrainingArguments(
    output_dir=f'./results_{dataset_name}_{epochs}_{batch_size_train}',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size_train,
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

# Metrics
bleu = evaluate.load("bleu")

def calculate_metrics(eval_pred):
    predictions, references = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # Convert predictions to tensor
    predictions = torch.tensor(predictions)
    predicted_ids = torch.argmax(predictions, dim=-1)
    
    decoded_preds = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
    decoded_refs = tokenizer.batch_decode(references, skip_special_tokens=True)
    
    # BLEU score
    bleu_score = bleu.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_refs])

    return {"bleu": bleu_score["bleu"]}

# Trainer setup
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator,
    compute_metrics=calculate_metrics
)
# Train the model
trainer.train()

# Save the model
path = f'model/bart_coba{num}-{epoch}-{batch_size}'
model.save_pretrained(path)
tokenizer.save_pretrained(path)
generation_config.save_pretrained(path)

# Evaluate model
eval_results = trainer.evaluate()

# Print evaluation results, including accuracy
print(f"Evaluation results: {eval_results}")
logging.info(f"Model: {path}")
logging.info(f"Evaluation results: {eval_results}")
logging.info("------------------------------------------\n")