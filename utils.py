import logging
import os
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig
import numpy as np

# Logging configuration
def logging_config(log_dir, log_filename):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=f'{log_dir}/{log_filename}',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Define the BartGenerator class
class BartGenerator:
    def init(self, model_path):
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model_path = model_path  # Store the model path
        
        # Load or create a GenerationConfig
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        
        # Ensure that necessary tokens are set in the generation config
        self.generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id or self.tokenizer.bos_token_id
        self.generation_config.bos_token_id = self.generation_config.bos_token_id or self.tokenizer.bos_token_id

        # Print statements to confirm the IDs
        print(f"decoder_start_token_id is set to: {self.generation_config.decoder_start_token_id}")
        print(f"bos_token_id is set to: {self.generation_config.bos_token_id}")
        logging.info(f"decoder_start_token_id is set to: {self.generation_config.decoder_start_token_id}")
        logging.info(f"bos_token_id is set to: {self.generation_config.bos_token_id}")

    def generate_answer(self, question, max_length=160):  # Adjusted max_length
        inputs = self.tokenizer(question, return_tensors='pt')
        
        # Use the generation configuration directly from the model config
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
    def init(self, inputs, targets, tokenizer, max_length=160):  # Adjusted max_length
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def len(self):
        return len(self.inputs)

    def getitem(self, idx):
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