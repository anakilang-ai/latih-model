# training.py

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import csv

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    def filter_valid_rows(row):
        return len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""

    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        filtered_rows = [row for row in reader if filter_valid_rows(row)]

    df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
    print(f"Number of valid rows: {len(df)}")

    label_encoder = LabelEncoder()
    df['encoded_answer'] = label_encoder.fit_transform(df['answer'])

    return df, label_encoder

def train_model(train_loader, val_loader, model, optimizer, device):
    model = model.to(device)
    model.train()

    for epoch in range(5):  # Set the number of epochs
        print(f"Epoch {epoch + 1}")
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        print(f"Training Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                val_correct_predictions += (predictions == labels).sum().item()
                val_total_predictions += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct_predictions / val_total_predictions
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    # Save model
    model_save_path = "./model"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    file_path = 'rf1.csv'
    df, label_encoder = load_data(file_path)
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_))
    
    # Prepare datasets and dataloaders
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['question'].tolist(),
        df['encoded_answer'].tolist(),
        test_size=0.3,
        random_state=42
    )
    
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_len=128)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_len=128)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_model(train_loader, val_loader, model, optimizer, device)
