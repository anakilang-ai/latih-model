import pandas as pd
import csv
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

# Load dataset
df = pd.read_csv('lar-clean.csv', delimiter='|', names=['question', 'answer'], encoding='utf-8', quoting=csv.QUOTE_NONE)

# Create binary labels (0 or 1) for demonstration purposes
df['label'] = df.index % 2  # Example: Use indexes as temporary labels

# Split the data into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load tokenizer and model
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Preprocess the data
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

# Tokenize datasets
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

# Initialize the Trainer
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

# Function to make predictions using the trained model
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

# Example usage for predictions with the trained model
if __name__ == "__main__":
    question = "cara minta transkrip nilai"
    answer = "Yth. Kepala Bagian Akademik Universitas XYZ di Tempat. Dengan hormat, Mahasiswa Universitas XYZ, Nama saya [Nama], dengan NIM [NIM]. Saya ingin meminta transkrip nilai semester [semester] yang telah saya tempuh. Demikian surat permohonan ini saya sampaikan, atas perhatian dan kerjasamanya saya ucapkan terima kasih. Hormat saya [Nama]"

    predicted_class = predict(question, answer)
    print(f"Pertanyaan: {question}")
    print(f"Jawaban: {answer}")
    print(f"Kelas Prediksi: {predicted_class}")
