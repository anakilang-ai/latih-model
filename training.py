# Import necessary libraries
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Define the path to the saved model and tokenizer
model_directory = "./model"
tokenizer = RobertaTokenizer.from_pretrained(model_directory)
model = RobertaForSequenceClassification.from_pretrained(model_directory)

# Example text for classification
sample_text = "Saya sangat senang dengan layanan yang ada saat ini!"

# Tokenize the input text
tokens = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)

# Perform prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

#prediksi label (0/1)
predicted_class = torch.argmax(logits, dim=1).item()

#Menampilkan Hasil
label_map = {0: "Negatif", 1: "Positif"}
print(f"Teks: {text}")
print(f"Prediksi Sentimen: {label_map[predicted_class]}")