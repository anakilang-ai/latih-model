import pandas as pd
import numpy as np
import logging
import csv
import torch
import evaluate
import os
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, GenerationConfig, AdamW, get_linear_schedule_with_warmup, EarlyStoppingCallback
from sklearn.model_selection import train_test_split as tts
from textaugment import EDA
from utils import QADataset, logging_config

# Logging configuration
logging_config('log_model', 'training.log')

# Function to filter valid rows
def filter_valid_rows(row):
    return len(row) == 2 and all(row)

# Load the dataset
num = 'dataset-kelas'
filtered_rows = []
with open(f'{num}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|', quoting=csv.QUOTE_NONE)
    for row in reader:
        if filter_valid_rows(row):
            filtered_rows.append(row)

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])

# Split dataset into training and test sets
train_df, test_df = tts(df, test_size=0.2, random_state=42)

# Reset index to ensure continuous indexing
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Prepare the dataset
model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)

# Combine question and answer into a single string for training
inputs_train = train_df['question'].tolist()
targets_train = train_df['answer'].tolist()

inputs_test = test_df['question'].tolist()
targets_test = test_df['answer'].tolist()

# Data augmentation
augmenter = EDA()
augmented_data = []
for question, answer in zip(inputs_train, targets_train):
    augmented_question = augmenter.synonym_replacement(question)
    augmented_answer = augmenter.synonym_replacement(answer)
    augmented_data.append((augmented_question, augmented_answer))

inputs_train_augmented, targets_train_augmented = zip(*augmented_data)
inputs_train += list(inputs_train_augmented)
targets_train += list(targets_train_augmented)

dataset_train = QADataset(inputs_train, targets_train, tokenizer, max_length=256)
dataset_test = QADataset(inputs_test, targets_test, tokenizer, max_length=256)

# Load model
model = BartForConditionalGeneration.from_pretrained(model_name)

# Define data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# epoch size and batchsize levels
epoch = 10  # Mengurangi jumlah epoch untuk eksperimen awal
batch_size = 8  # Menyesuaikan batch size untuk menghindari out of memory

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
    save_steps=200,  # Menyimpan checkpoint lebih sering
    save_total_limit=5,  # Menyimpan lebih banyak checkpoint
    fp16=True,
    evaluation_strategy="steps",  # Lakukan evaluasi setiap beberapa langkah
    eval_steps=200,  # Interval evaluasi
    gradient_accumulation_steps=2,  # Mengakumulasikan gradien selama 2 step untuk mensimulasikan batch size yang lebih besar
    dataloader_num_workers=4,  # Memanfaatkan lebih banyak CPU untuk loading data
)

# Define generation config
generation_config = GenerationConfig(
    early_stopping=True,
    num_beams=5, 
    no_repeat_ngram_size=0,
    forced_bos_token_id=0,
    forced_eos_token_id=2,
    max_length=256,  
    bos_token_id=0,
    decoder_start_token_id=2
)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=160, 
    num_training_steps=len(dataset_train) * epoch // batch_size
)

# Load metrics
bleu_metric = evaluate.load("bleu")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = torch.tensor(logits)
    predictions = torch.argmax(logits, dim=-1)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    bleu = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    logging.info(f"BLEU score: {bleu['bleu']}")
    
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
    compute_metrics=compute_metrics,
    optimizers=(optimizer, scheduler),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # Early stopping
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