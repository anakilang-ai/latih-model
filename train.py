import pandas as pd
import csv
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

# Set random seed for reproducibility
set_seed(42)

# Function to load and prepare data
def load_and_prepare_data(file_path, delimiter='|'):
    df = pd.read_csv(file_path, delimiter=delimiter, names=['question', 'answer'], encoding='utf-8', quoting=csv.QUOTE_NONE)
    df['label'] = df.index % 2  # Using indexes as temporary labels for binary classification
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    return train_df, test_df

# Function to create Hugging Face datasets
def create_datasets(train_df, test_df):
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    return train_dataset, test_dataset

# Function to preprocess data
def preprocess_function(examples):
    inputs = tokenizer(
        examples['question'], 
        examples['answer'], 
        truncation=True, 
        padding='max_length', 
        max_length=512
    )
    inputs['labels'] = examples['label']
    return inputs

# Load and prepare data
train_df, test_df = load_and_prepare_data('lar-clean.csv')

# Load tokenizer and model
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Create datasets
train_dataset, test_dataset = create_datasets(train_df, test_df)

# Preprocess datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Create data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

# Function to make predictions
def predict(question, answer):
    inputs = tokenizer(
        question,
        answer,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Example using predictions
question = "cara minta transkrip nilai"
answer = "Yth. Kepala Bagian Akademik Universitas XYZdi TempatDengan hormat,Mahasiswa Universitas XYZ, Nama saya [Nama], dengan NIM [NIM]. Saya ingin meminta transkrip nilai semester [semester] yang telah saya tempuh.Demikian surat permohonan ini saya sampaikan, atas perhatian dan kerjasamanya saya ucapkan terima kasih.Hormat saya [Nama]"

predicted_class = predict(question, answer)
print(f"Pertanyaan: {question}")
print(f"Jawaban: {answer}")
print(f"Kelas Prediksi: {predicted_class}")
