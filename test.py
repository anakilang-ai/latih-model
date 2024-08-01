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
            answer = generator.generate_answer(question)
            print(f"Jawaban: {answer}")

            # Log the result
            logging.info(f"Model: {generator.model_path}")
            logging.info(f"Pertanyaan: {question}")
            logging.info(f"Jawaban: {answer}")
            logging.info("------------------------------------------\n")
        except ValueError as e:
            print(e)

if name == "main":
    main()