# test1.py

import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

def load_data_and_encoder(file_path):
    # Membaca data dan mengkodekan label
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""]

    df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
    label_encoder = LabelEncoder()
    label_encoder.fit(df['answer'])
    
    return df, label_encoder

def preprocess_data(df, tokenizer):
    # Mempersiapkan data untuk prediksi
    input_ids = []
    attention_masks = []
    for question in df['question']:
        encoded = tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    
    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)
    
    return input_ids, attention_masks

def predict(question, model, tokenizer, label_encoder):
    # Fungsi untuk memprediksi jawaban berdasarkan input pertanyaan
    encoded = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )

    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    logits = model(input_ids, attention_mask=attention_mask).logits
    predicted_label_id = tf.argmax(logits, axis=1).numpy()[0]
    predicted_label = label_encoder.inverse_transform([predicted_label_id])[0]
    
    return predicted_label

def evaluate_model(model, tokenizer, label_encoder, file_path):
    # Evaluasi akurasi model pada data uji
    df_test, _ = load_data_and_encoder(file_path)
    df_test['encoded_answer'] = label_encoder.transform(df_test['answer'])
    
    input_ids_test, attention_masks_test = preprocess_data(df_test, tokenizer)
    labels_test = tf.constant(df_test['encoded_answer'].values)

    logits_test = model(input_ids_test, attention_mask=attention_masks_test).logits
    preds_test = tf.argmax(logits_test, axis=1, output_type=tf.int32)
    
    test_accuracy_metric = tf.keras.metrics.Accuracy()
    test_accuracy_metric.update_state(labels_test, preds_test)
    
    test_accuracy = test_accuracy_metric.result()
    print(f"Test accuracy: {test_accuracy:.4f}")

def main():
    # Load model dan tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta_model')
    model = TFRobertaForSequenceClassification.from_pretrained('roberta_model')
    
    # Load encoder label
    _, label_encoder = load_data_and_encoder('rf1.csv')
    
    # Evaluasi model
    evaluate_model(model, tokenizer, label_encoder, 'rf1.csv')
    
    # Loop interaktif untuk prediksi pengguna
    while True:
        question = input("Masukkan pertanyaan (atau 'keluar' untuk keluar): ")
        if question.lower() == 'keluar':
            break
        answer = predict(question, model, tokenizer, label_encoder)
        print(f"Jawaban yang Diprediksi: {answer}")

if __name__ == "__main__":
    main()
