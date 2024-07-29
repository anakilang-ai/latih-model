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

dataset_train = QADataset(inputs_train, targets_train, tokenizer, max_length=160)
dataset_test = QADataset(inputs_test, targets_test, tokenizer, max_length=160)

# Load model
model = BartForConditionalGeneration.from_pretrained(model_name)

# Define data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# epoch size and batchsize levels
epoch = 20
batch_size = 10

# Define training arguments
training_args = TrainingArguments(
    output_dir=f'./result/results_coba{num}-{epoch}-{batch_size}',
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