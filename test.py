from unittest.mock import patch
import unittest
import logging
from your_module import BartGenerator, generate_answer

class TestAnswerGeneration(unittest.TestCase):
    
    def setUp(self):
        self.generator = BartGenerator('path_to_model')
    
    @patch('logging.info')
    def test_generate_answer_logging_order(self, mock_log_info):
        # Simulate main function behavior
        question = "What is the capital of France?"
        expected_answer = "Paris"  # Adjust this to your expected output
        
        # Mock the generate_answer method
        with patch.object(self.generator, 'generate_answer', return_value=expected_answer):
            answer = self.generator.generate_answer(question)
            
            # Assert logging in the order of calls
            mock_log_info.assert_called_with(f"Model: {self.generator.model_path}")
            mock_log_info.assert_called_with(f"Pertanyaan: {question}")
            mock_log_info.assert_called_with(f"Jawaban: {answer}")
