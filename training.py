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
    predictions = model(**tokens)
    logits = predictions.logits

# Determine the predicted class (0/1)
predicted_label = torch.argmax(logits, dim=1).item()

# Define the label mapping
label_mapping = {0: "Negatif", 1: "Positif"}

# Display the results
print(f"Teks: {sample_text}")
print(f"Prediksi Sentimen: {label_mapping[predicted_label]}")
