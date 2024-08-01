import logging
import os
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig

# Logging configuration
def logging_config(log_dir, log_filename):
    """Set up logging configuration."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=os.path.join(log_dir, log_filename),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Define the BartGenerator class
class BartGenerator:
    def __init__(self, model_path):
        """
        Initialize the BartGenerator with the specified model path.

        Args:
            model_path (str): The path to the pre-trained model.
        """
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model_path = model_path

        # Load or create a GenerationConfig
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        
        # Ensure that necessary tokens are set in the generation config
        if self.generation_config.decoder_start_token_id is None:
            self.generation_config.decoder_start_token_id = self.tokenizer.bos_token_id
        if self.generation_config.bos_token_id is None:
            self.generation_config.bos_token_id = self.tokenizer.bos_token_id

        # Log configuration
        logging.info(f"decoder_start_token_id is set to: {self.generation_config.decoder_start_token_id}")
        logging.info(f"bos_token_id is set to: {self.generation_config.bos_token_id}")

    def generate_answer(self, question, max_length=160):
        """
        Generate an answer to a given question.

        Args:
            question (str): The input question.
            max_length (int): The maximum length of the generated response.

        Returns:
            str: The generated answer.
        """
        inputs = self.tokenizer(question, return_tensors='pt')
        
        # Generate output using model
        outputs = self.model.generate(
            inputs['input_ids'],
            early_stopping=True,
            num_beams=5,
            no_repeat_ngram_size=2,  # Avoid repeating n-grams
            max_length=max_length,
            bos_token_id=self.generation_config.bos_token_id,
            decoder_start_token_id=self.generation_config.decoder_start_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Define the QADataset class
class QADataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=160):
        """
        Initialize the QADataset with inputs, targets, and tokenizer.

        Args:
            inputs (list of str): List of input questions.
            targets (list of str): List of target answers.
            tokenizer (BartTokenizer): The tokenizer used to preprocess the data.
            max_length (int): The maximum sequence length for padding/truncation.
        """
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing input_ids, attention_mask, and labels tensors.
        """
        input_encoding = self.tokenizer(
            self.inputs[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            self.targets[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = input_encoding.input_ids.squeeze().numpy()
        attention_mask = input_encoding.attention_mask.squeeze().numpy()
        labels = target_encoding.input_ids.squeeze().numpy()

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
