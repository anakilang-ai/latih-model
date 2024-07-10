# import
import torch
from transformers import RobertaTokenizerS RobertaForSequenceClassification

# Loading model and tokenizer from 'model' Folders
model_save_path = "./model"
tokenizer = RobertaTokenizer.from_pretrained(model_save_path)
model = RobertaForSequenceClassification.from_pretrained(model_save_path)

#Example of text to be classified
text = "Saya sangat senang dengan layanan yang ada! "

#Text Tokenization
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Make Predictions
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# label prediction (0/1)
predicted_class = torch.argmax(logits, dim=1).item()

#Displaying Results
label_map = {0: "Negatif", 1: "Positif"}
print(f"Teks: {text}")
print(f"Prediksi Sentimen: {label_map[predicted_class]}")
