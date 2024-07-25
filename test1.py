import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import csv

# Define the path to the pre-trained model
model_path = 'path/to/your/pretrained/roberta_model'  # Update with the correct path or model name

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = TFRobertaForSequenceClassification.from_pretrained(model_path)

# Load and preprocess label encoder data
print("Loading and preprocessing label encoder data...")
with open('rf1.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    # Filter rows to remove any with missing data
    filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() and row[1].strip()]

# Create a DataFrame and fit the LabelEncoder
df = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
label_encoder = LabelEncoder()
label_encoder.fit(df['answer'])

# Function to make predictions
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

    # Perform prediction
    try:
        logits = model(input_ids, attention_mask=attention_mask).logits
        # Get predicted label ID
        predicted_label_id = tf.argmax(logits, axis=1).numpy()[0]
        # Decode label ID to original label
        predicted_label = label_encoder.inverse_transform([predicted_label_id])[0]
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None
    
    return predicted_label
# Evaluate accuracy on test set
print("Evaluating model accuracy on test set...")
test_accuracy_metric = tf.keras.metrics.Accuracy()

# Read test data and preprocess
with open('rf1.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='|')
    filtered_rows = [row for row in reader if len(row) == 2 and row[0].strip() and row[1].strip()]

df_test = pd.DataFrame(filtered_rows, columns=['question', 'answer'])
df_test['encoded_answer'] = label_encoder.transform(df_test['answer'])

# Prepare test data for model
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

# Perform predictions and compute accuracy
print("Computing predictions and accuracy...")

# Get model logits for test data
logits_test = model(input_ids_test, attention_mask=attention_masks_test).logits

# Convert logits to predictions
preds_test = tf.argmax(logits_test, axis=1, output_type=tf.int32)

# Calculate and update accuracy
test_accuracy_metric.update_state(labels_test, preds_test)

# Compute final accuracy result
test_accuracy = test_accuracy_metric.result().numpy()  # Convert tensor to numpy scalar for better readability
print(f"Test accuracy: {test_accuracy:.4f}")


# Interactive loop for user input and predictions
print("Interactive prediction loop. Type 'exit' to quit.")

while True:
    try:
        # Get user input
        question = input("Enter a question (or type 'exit' to quit): ")
        
        # Exit condition
        if question.lower() == 'exit':
            print("Exiting the prediction loop. Goodbye!")
            break
        
        # Make prediction
        answer = predict(question)
        
        # Display the prediction result
        print(f"Predicted Answer: {answer}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
