# import train
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

#Create Hugging Face dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

#Loading tokenizer and model
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

#Data preprocessing
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

#Dataset tokenization
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

#Data collator
data_collator = DataCollatorWithPadding(tokenizer)

#The training argument
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

#Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

#Training the model
trainer.train()

#Saves the model and tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

#Example of use for predictions with pre-trained models
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

#Example of using Predictions
question = "cara minta transkrip nilai"
answer = "Yth. Kepala Bagian Akademik Universitas XYZdi TempatDengan hormat,Mahasiswa Universitas XYZ, Nama saya [Nama], dengan NIM [NIM]. Saya ingin meminta transkrip nilai semester [semester] yang telah saya tempuh.Demikian surat permohonan ini saya sampaikan, atas perhatian dan kerjasamanya saya ucapkan terima kasih.Hormat saya [Nama]"

predicted_class = predict(question, answer)
print(f"Pertanyaan: {question}")
print(f"Jawaban: {answer}")
print(f"Kelas Prediksi: {predicted_class}")
