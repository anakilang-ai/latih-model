import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

# Load tokenizer & model
tokenizer = RobertaTokenizer.from_pretrained('roberta_model')
model = TFRobertaForSequenceClassification.from_pretrained('roberta_model')

# Function to load and process CSV file
def load_and_process_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""]
    df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
    return df

# Load Encoder label
df = load_and_process_csv('rf1.csv')
label_encoder = LabelEncoder()
label_encoder.fit(df['answer'])

# function to make pedictions 
def predict(question):
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

# Calculate accuracy on test set
def calculate_test_accuracy(file_path):
    test_accuracy_metric = tf.keras.metrics.Accuracy()
    df_test = load_and_process_csv(file_path)
    df_test['encoded_answer'] = label_encoder.transform(df_test['answer'])

    input_ids_test = []
    attention_masks_test = []
    for question in df_test['question']:
        input_ids, attention_mask = encode_question(question)
        input_ids_test.append(input_ids)
        attention_masks_test.append(attention_mask)

    input_ids_test = tf.concat(input_ids_test, axis=0)
    attention_masks_test = tf.concat(attention_masks_test, axis=0)
    labels_test = tf.constant(df_test['encoded_answer'].values)

logits_test = model(input_ids_test, attention_mask=attention_masks_test).logits
preds_test = tf.argmax(logits_test, axis=1, output_type=tf.int32)
test_accuracy_metric.update_state(labels_test, preds_test)

test_accuracy = test_accuracy_metric.result()
print(f"Test accuracy: {test_accuracy:.4f}")

# Loop interaktif untuk input & prediksi pengguna
while True:
    question = input("Enter a question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    answer = predict(question)
    print(f"Predicted Answer: {answer}")
