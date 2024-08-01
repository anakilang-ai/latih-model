# Save the model, tokenizer, and generation configuration
model_save_path = f'model/bart_coba{num}-{epoch}-{batch_size}'

# Ensure the directory exists
os.makedirs(model_save_path, exist_ok=True)

try:
    # Save the model
    model.save_pretrained(model_save_path)
    logging.info(f"Model saved to: {model_save_path}/pytorch_model.bin")

    # # Save the model, tokenizer, and generation configuration
model_save_path = f'model/bart_coba{num}-{epoch}-{batch_size}'

# Ensure the directory exists
os.makedirs(model_save_path, exist_ok=True)

try:
    # Save the model
    model.save_pretrained(model_save_path)
    logging.info(f"Model saved to: {model_save_path}/pytorch_model.bin")

    # Save the tokenizer
    tokenizer.save_pretrained(model_save_path)
    logging.info(f"Tokenizer saved to: {model_save_path}/tokenizer_config.json")

    # Save the generation configuration
    generation_config.save_pretrained(model_save_path)
    logging.info(f"Generation configuration saved to: {model_save_path}/generation_config.json")

except Exception as e:
    logging.error(f"An error occurred while saving the model or tokenizer: {str(e)}")
    print(f"An error occurred while saving the model or tokenizer: {str(e)}")

    tokenizer.save_pretrained(model_save_path)
    logging.info(f"Tokenizer saved to: {model_save_path}/tokenizer_config.json")

    # Save the generation configuration
    generation_config.save_pretrained(model_save_path)
    logging.info(f"Generation configuration saved to: {model_save_path}/generation_config.json")

except Exception as e:
    logging.error(f"An error occurred while saving the model or tokenizer: {str(e)}")
    print(f"An error occurred while saving the model or tokenizer: {str(e)}")
