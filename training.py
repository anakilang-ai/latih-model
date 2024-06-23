import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import torch
import csv

# Load the dataset
df = pd.read_csv('dataset/0.csv', delimiter='|', names=['question', 'answer'], encoding='utf-8', quoting=csv.QUOTE_NONE)

# Prepare the dataset
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Combine question and answer into a single string for training
inputs = df['question'] + tokenizer.eos_token + df['answer']

class QADataset(torch.utils.data.Dataset):
    def __init__(self, inputs, tokenizer, max_length=64):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        try:
            encodings = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
            input_ids = encodings['input_ids'].squeeze()
            attention_mask = encodings['attention_mask'].squeeze()
        except Exception as e:
            print(f"Error processing input_text ({idx}): {input_text}")
            raise e
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}

dataset = QADataset(inputs, tokenizer)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',         
    num_train_epochs=3,              
    per_device_train_batch_size=2,  
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
)

# Create Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=dataset,         
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('gpt2_model')
tokenizer.save_pretrained('gpt2_model')