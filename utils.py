import logging
import os
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig
import numpy as np

# Logging setup
def configure_logging(log_directory, log_file):
    if not os.path.isdir(log_directory):
        os.makedirs(log_directory)

    logging.basicConfig(
        filename=os.path.join(log_directory, log_file),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Define the BartGenerator class
class BartGenerator:
    def __init__(self, model_directory):
        self.tokenizer = BartTokenizer.from_pretrained(model_directory)
        self.model = BartForConditionalGeneration.from_pretrained(model_directory)
        self.model_directory = model_directory  # Save the model directory
        
        # Load or initialize a GenerationConfig
        self.generation_config = GenerationConfig.from_pretrained(model_directory)
        
        # Ensure that the necessary tokens are set in the generation configuration
        self.generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id or self.tokenizer.bos_token_id
        self.generation_config.bos_token_id = self.generation_config.bos_token_id or self.tokenizer.bos_token_id

        # Log statements to confirm the token IDs
        print(f"decoder_start_token_id: {self.generation_config.decoder_start_token_id}")
        print(f"bos_token_id: {self.generation_config.bos_token_id}")
        logging.info(f"decoder_start_token_id: {self.generation_config.decoder_start_token_id}")
        logging.info(f"bos_token_id: {self.generation_config.bos_token_id}")

    def generate_response(self, prompt, max_length=160):  # Adjusted max_length
        inputs = self.tokenizer(prompt, return_tensors='pt')
        
        # Generate using the configuration from the model
        outputs = self.model.generate(
            inputs['input_ids'],
            early_stopping=True,
            num_beams=5, 
            no_repeat_ngram_size=0,
            forced_bos_token_id=0,
            forced_eos_token_id=2,
            max_length=160,  
            bos_token_id=0,
            decoder_start_token_id=2
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define the QADataset class
class QADataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=160):  # Adjusted max_length
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_encoding = self.tokenizer(self.inputs[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors='pt')
        target_encoding = self.tokenizer(self.targets[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors='pt')
        
        input_ids = input_encoding.input_ids.squeeze().numpy()
        attention_mask = input_encoding.attention_mask.squeeze().numpy()
        labels = target_encoding.input_ids.squeeze().numpy()

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }