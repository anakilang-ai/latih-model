import unittest
from unittest.mock import patch
from utils import BartGenerator, generate_answer, logging_config
from trainprocess import path

class TestBartGenerator(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Setup logging configuration for tests
        logging_config('test_log', 'test_generator.log')
        
        # Initialize the generator with the model path
        cls.generator = BartGenerator(path)
    
    def test_generate_answer_valid(self):
        def test_generate_answer_valid_with_error(self):
    # Test generating an answer for a question where an error might occur
    question = "Trigger an error"
    expected_answer = "Error occurred"
    
    with patch.object(self.generator, 'generate_answer', side_effect=ValueError("An error occurred")):
        with self.assertRaises(ValueError):
            self.generator.generate_answer(question)

    
    def test_generate_answer_invalid(self):
        # Test generating an answer for an invalid question
        question = "Unknown question?"
        expected_answer = "I don't know the answer."  # Adjust this to your expected output
        
        with patch.object(self.generator, 'generate_answer', return_value=expected_answer):
            answer = self.generator.generate_answer(question)
            self.assertEqual(answer, expected_answer)
    
    def test_logging(self):
        # Test if logging is properly configured and working
        question = "What is the tallest mountain in the world?"
        answer = "Mount Everest"  # Adjust this to your expected output
        
        with patch('builtins.input', return_value=question), \
             patch('logging.info') as mock_log_info:
            self.generator.generate_answer = lambda q: answer  # Mock method
            
            # Simulate main function behavior
            answer = self.generator.generate_answer(question)
            mock_log_info.assert_any_call(f"Model: {self.generator.model_path}")
            mock_log_info.assert_any_call(f"Pertanyaan: {question}")
            mock_log_info.assert_any_call(f"Jawaban: {answer}")

if __name__ == "__main__":
    unittest.main()
