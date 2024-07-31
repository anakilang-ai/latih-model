import logging
import os
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig

# Setup logging configuration
def configure_logging(directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    logging.basicConfig(
        filename=os.path.join(directory, filename),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Bart Generator class definition
class BartAnswerGenerator:
    def __init__(self, model_directory):
        self.tokenizer = BartTokenizer.from_pretrained(model_directory)
        self.model = BartForConditionalGeneration.from_pretrained(model_directory)
        self.model_directory = model_directory  # Save the model directory
        
        # Initialize or load GenerationConfig
        self.generation_config = GenerationConfig.from_pretrained(model_directory)
        
        # Set necessary tokens in the generation config if not already set
        self.generation_config.decoder_start_token_id = self.generation_config.decoder_start_token_id or self.tokenizer.bos_token_id
        self.generation_config.bos_token_id = self.generation_config.bos_token_id or self.tokenizer.bos_token_id

        # Log and print token IDs
        print(f"decoder_start_token_id: {self.generation_config.decoder_start_token_id}")
        print(f"bos_token_id: {self.generation_config.bos_token_id}")
        logging.info(f"decoder_start_token_id: {self.generation_config.decoder_start_token_id}")
        logging.info(f"bos_token_id: {self.generation_config.bos_token_id}")

    def generate_response(self, query, max_length=160):  # Set max_length
        inputs = self.tokenizer(query, return_tensors='pt')
        
        # Generate the response using the configured generation settings
        outputs = self.model.generate(
            inputs['input_ids'],
            early_stopping=True,
            num_beams=5,
            no_repeat_ngram_size=0,
            forced_bos_token_id=0,
            forced_eos_token_id=2,
            max_length=max_length,  
            bos_token_id=0,
            decoder_start_token_id=2
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define the QADataset class
class QADataset(Dataset):
    def _init_(self, inputs, targets, tokenizer, max_length=160):  # Adjusted max_length
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _len_(self):
        return len(self.inputs)

    def _getitem_(self, idx):
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