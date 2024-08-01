import os
import logging

# Define the save path for the model, tokenizer, and generation configuration
model_save_path = f'model/bart_coba{num}-{epoch}-{batch_size}'

# Ensure the save directory exists
os.makedirs(model_save_path, exist_ok=True)

def save_component(component, path, description):
    try:
        # Save the component
        component.save_pretrained(path)
        
        # Check if the file was saved
        if os.path.exists(path):
            logging.info(f"{description} successfully saved to: {path}")
        else:
            logging.warning(f"{description} file not found at: {path}")
    
    except Exception as e:
        # Detailed error logging
        logging.error(f"Error saving {description} to {path}: {str(e)}")
        print(f"Error saving {description} to {path}: {str(e)}")

# Define file paths
model_file_path = os.path.join(model_save_path, 'pytorch_model.bin')
tokenizer_path = os.path.join(model_save_path, 'tokenizer')
generation_config_path = os.path.join(model_save_path, 'generation_config')

# Save the model
save_component(model, model_file_path, 'Model')

# Save the tokenizer (tokenizer.save_pretrained already saves the tokenizer config)
save_component(tokenizer, tokenizer_path, 'Tokenizer')

# Save the generation configuration
save_component(generation_config, generation_config_path, 'Generation configuration')

# Optional: Verify the directory contents
if os.path.isdir(model_save_path):
    saved_files = os.listdir(model_save_path)
    logging.info(f"Directory {model_save_path} contains: {saved_files}")
else:
    logging.warning(f"Save directory {model_save_path} does not exist.")
