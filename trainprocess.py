import pandas as pd
import numpy as np
import logging
import csv
import torch
import evaluate
import os
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, GenerationConfig
from sklearn.model_selection import train_test_split
from utils import QADataset, logging_config

# Konfigurasi logging
logging_config('log_model', 'training.log')

# Fungsi untuk memfilter baris yang valid
def is_valid_row(row):
    return len(row) == 2 and all(row)

# Memuat dataset
dataset_name = 'dataset-kelas'
valid_rows = []
with open(f'{dataset_name}.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|', quoting=csv.QUOTE_NONE)
    for row in reader:
        if is_valid_row(row):
            valid_rows.append(row)

df = pd.DataFrame(valid_rows, columns=['question', 'answer'])

# Membagi dataset menjadi set pelatihan dan pengujian
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Reset indeks untuk memastikan indeks kontinu
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Mempersiapkan dataset
model_name = 'facebook/bart-base'
tokenizer = BartTokenizer.from_pretrained(model_name)

# Menggabungkan pertanyaan dan jawaban menjadi satu string untuk pelatihan
train_inputs = train_df['question'].tolist()
train_targets = train_df['answer'].tolist()

test_inputs = test_df['question'].tolist()
test_targets = test_df['answer'].tolist()

train_dataset = QADataset(train_inputs, train_targets, tokenizer, max_length=160)
test_dataset = QADataset(test_inputs, test_targets, tokenizer, max_length=160)

# Memuat model
model = BartForConditionalGeneration.from_pretrained(model_name)

# Mendefinisikan data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
)

# Ukuran epoch dan batch size
num_epochs = 20
train_batch_size = 10

# Mendefinisikan argumen pelatihan
training_args = TrainingArguments(
    output_dir=f'./result/results_{dataset_name}_{num_epochs}_{train_batch_size}',
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