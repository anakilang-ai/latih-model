import os
import logging

# Define the save path for the model, tokenizer, and generation configuration
model_save_path = f'model/bart_coba{num}-{epoch}-{batch_size}'

# Ensure the save directory exists
os.makedirs(model_save_path, exist_ok=True)

def save_component(component, path, description):
    try:
        component.save_pretrained(path)
        if os.path.exists(path):
            logging.info(f"{description} successfully saved to: {path}")
        else:
            logging.warning(f"{description} file not found after saving: {path}")
    except Exception as e:
        logging.error(f"Error saving {description}: {str(e)}")
        print(f"Error saving {description}: {str(e)}")

# Save the model, tokenizer, and generation configuration
model_path = os.path.join(model_save_path, 'pytorch_model.bin')
tokenizer_path = os.path.join(model_save_path, 'tokenizer_config.json')
generation_config_path = os.path.join(model_save_path, 'generation_config.json')

save_component(model, model_path, 'Model')
save_component(tokenizer, tokenizer_path, 'Tokenizer')
save_component(generation_config, generation_config_path, 'Generation configuration')

# Optional: Verify the directory contents
if os.path.isdir(model_save_path):
    saved_files = os.listdir(model_save_path)
    logging.info(f"Directory {model_save_path} contains: {saved_files}")
else:
    logging.warning(f"Save directory {model_save_path} does not exist.")
