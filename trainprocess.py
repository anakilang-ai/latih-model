# Define the save path for the model, tokenizer, and generation configuration
model_save_path = f'model/bart_coba{num}-{epoch}-{batch_size}'

# Ensure the save directory exists
os.makedirs(model_save_path, exist_ok=True)

try:
    # Save the model
    model_path = os.path.join(model_save_path, 'pytorch_model.bin')
    model.save_pretrained(model_save_path)
    logging.info(f"Model saved to: {model_path}")

    # Save the tokenizer
    tokenizer_path = os.path.join(model_save_path, 'tokenizer_config.json')
    tokenizer.save_pretrained(model_save_path)
    logging.info(f"Tokenizer saved to: {tokenizer_path}")

    # Save the generation configuration
    generation_config_path = os.path.join(model_save_path, 'generation_config.json')
    generation_config.save_pretrained(model_save_path)
    logging.info(f"Generation configuration saved to: {generation_config_path}")

except Exception as e:
    # Log the error with detailed information
    logging.error(f"An error occurred while saving the model, tokenizer, or generation configuration: {str(e)}")
    print(f"An error occurred while saving the model, tokenizer, or generation configuration: {str(e)}")
