import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Loading model and tokenizer from "models" folder
model_save_path = "./model"
tokenizer = RobertaTokenizer.from_pretrained(model_save_path)
model = RobertaForSequenceClassification.from_pretrained(model_save_path)

# Example of text to be classified
text = "Saya sangat senang dengan layanan ini!"

# Text Tokenization
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Make predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get label prediction (0 or 1)
predicted_class = torch.argmax(logits, dim=1).item()

# Menampilkan Hasil
label_map = {0: "Negatif", 1: "Positif"}
print(f"Teks: {text}")
print(f"Prediksi Sentimen: {label_map[predicted_class]}")
