import os

# Define the save path for the model, tokenizer, and generation configuration
model_save_path = f'model/bart_coba{num}-{epoch}-{batch_size}'

# Ensure the save directory exists
os.makedirs(model_save_path, exist_ok=True)

try:
    # Save the model
    model_path = os.path.join(model_save_path, 'pytorch_model.bin')
    model.save_pretrained(model_save_path)
    if os.path.isfile(model_path):
        logging.info(f"Model saved successfully to: {model_path}")
    else:
        logging.warning(f"Model file not found after saving: {model_path}")

    # Save the tokenizer
    tokenizer_path = os.path.join(model_save_path, 'tokenizer_config.json')
    tokenizer.save_pretrained(model_save_path)
    if os.path.isfile(tokenizer_path):
        logging.info(f"Tokenizer saved successfully to: {tokenizer_path}")
    else:
        logging.warning(f"Tokenizer file not found after saving: {tokenizer_path}")

    # Save the generation configuration
    generation_config_path = os.path.join(model_save_path, 'generation_config.json')
    generation_config.save_pretrained(model_save_path)
    if os.path.isfile(generation_config_path):
        logging.info(f"Generation configuration saved successfully to: {generation_config_path}")
    else:
        logging.warning(f"Generation configuration file not found after saving: {generation_config_path}")

except Exception as e:
    # Log the error with detailed information
    logging.error(f"An error occurred while saving the model, tokenizer, or generation configuration: {str(e)}")
    print(f"An error occurred while saving the model, tokenizer, or generation configuration: {str(e)}")

# Optional: Verify the directory contents
if os.path.isdir(model_save_path):
    saved_files = os.listdir(model_save_path)
    logging.info(f"Directory {model_save_path} contains: {saved_files}")
else:
    logging.warning(f"Save directory {model_save_path} does not exist.")
