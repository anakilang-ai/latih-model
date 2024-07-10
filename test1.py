# import all
import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta_model')
model = TFRobertaForSequenceClassification.from_pretrained('roberta_model')

#Muat label Encoder
with open('rf1.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""]

df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
label_encoder = LabelEncoder()
label_encoder.fit(df['answer'])

# Function to Make Predictions
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

# Calculate accuracy On the Test Set
test_accuracy_metric = tf.keras.metrics.Accuracy()

# Uji modelnya
with open('rf1.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() != "" and row[1].strip() != ""]

df_test = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
df_test['encoded_answer'] = label_encoder.transform(df_test['answer'])

input_ids_test = []
attention_masks_test = []
for question in df_test['question']:
    encoded = tokenizer.encode_plus(
        question,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    input_ids_test.append(encoded['input_ids'])
    attention_masks_test.append(encoded['attention_mask'])

input_ids_test = tf.concat(input_ids_test, axis=0)
attention_masks_test = tf.concat(attention_masks_test, axis=0)
labels_test = tf.constant(df_test['encoded_answer'].values)

logits_test = model(input_ids_test, attention_mask=attention_masks_test).logits
preds_test = tf.argmax(logits_test, axis=1, output_type=tf.int32)
test_accuracy_metric.update_state(labels_test, preds_test)

test_accuracy = test_accuracy_metric.result()
print(f"Test accuracy: {test_accuracy:.4f}")

# Interactive loop to  user input & predict
while True:
    question = input("Enter a question (or 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    answer = predict(question)
    print(f"Predicted Answer: {answer}")
