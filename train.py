import pandas as pd
import csv
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch

# load dataset train
df = pd.read_csv('lar-clean.csv', delimiter='|', names=['question', 'answer'], encoding='utf-8', quoting=csv.QUOTE_NONE)

# Buat label biner (0 atau 1) dari data jawaban jika perlu
df['label'] = df.index % 2  # For example, using indexes as temporary labels

# Pisahkan data menjadi pelatihan dan pengujian
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# create hugging face dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

#Load tokenizer and model
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

#preprocessing data
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

#Dataset token
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# make Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# add The training argument
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# add Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
print("Starting training...")
trainer.train()

# Save the model and tokenizer
print("Saving model and tokenizer...")
model_save_path = "./model"
tokenizer_save_path = "./model"

# Save the model
model.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

# Save the tokenizer
tokenizer.save_pretrained(tokenizer_save_path)
print(f"Tokenizer saved to {tokenizer_save_path}")


# Example of use for predictions with pre-trained models
def predict(question, answer):
    # Tokenize inputs and prepare tensors for the model
    inputs = tokenizer(
        text=[question],  # Use text parameter for single strings
        text_pair=[answer],  # Pair question and answer
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors="pt"
    )
    
    # Ensure the model is in evaluation mode
    model.eval()

    # Perform prediction without tracking gradients
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract logits and determine the predicted class
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return predicted_class


# Example using predictions
question = "cara minta transkrip nilai"
answer = ("Yth. Kepala Bagian Akademik Universitas XYZ di Tempat. "
          "Dengan hormat, Mahasiswa Universitas XYZ, Nama saya [Nama], "
          "dengan NIM [NIM]. Saya ingin meminta transkrip nilai semester [semester] "
          "yang telah saya tempuh. Demikian surat permohonan ini saya sampaikan, "
          "atas perhatian dan kerjasamanya saya ucapkan terima kasih. "
          "Hormat saya [Nama]")

# Make predictions
predicted_class = predict(question, answer)

# Mapping of class labels to human-readable names (if needed)
label_names = {0: "Class 0", 1: "Class 1"}  # Update with actual class names if available

# Output results
print(f"Pertanyaan: {question}")
print(f"Jawaban: {answer}")
print(f"Kelas Prediksi: {label_names.get(predicted_class, 'Unknown')}")
