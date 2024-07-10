import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Memuat model dan tokenizer dari folder "model"
model_save_path = "./model"
tokenizer = RobertaTokenizer.from_pretrained(model_save_path)
model = RobertaForSequenceClassification.from_pretrained(model_save_path)

# Contoh teks yang akan diklasifikasikan
text = "Saya sangat senang dengan layanan ini!"

# Tokenisasi teks
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Melakukan prediksi
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Mendapatkan prediksi label (0 atau 1)
predicted_class = torch.argmax(logits, dim=1).item()

# Menampilkan hasil
label_map = {0: "Negatif", 1: "Positif"}
print(f"Teks: {text}")
print(f"Prediksi Sentimen: {label_map[predicted_class]}")
