import logging
import os
from utils import setup_logging, TextGenerator
from trainprocess import model_directory

# Configure logging
setup_logging('logs', 'test_generator.log')

def run_interactive_session():
    text_generator = TextGenerator(model_directory)

    while True:
        user_input = input("Enter your question (or type 'exit' to quit): ").strip()

        if user_input.lower() == 'exit':
            print("Exiting the program...")
            break

        try:
            response = text_generator.generate_text(user_input)
            print(f"Response: {response}")

            # Log the interaction
            logging.info(f"Model Directory: {text_generator.model_directory}")
            logging.info(f"Question: {user_input}")
            logging.info(f"Response: {response}")
            logging.info("------------------------------------------\n")
        except ValueError as error:
            print(error)

if __name__ == "__main__":
    run_interactive_session()