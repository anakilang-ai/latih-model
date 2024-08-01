import logging
import os
from utils import logging_config, BartGenerator
from trainprocess import path

# Configuring logging
logging_config('log_model', 'generator_test.log')

def main():
    # Initializing the generator
    generator = BartGenerator(path)

    while True:
        question = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ").strip()

        if question.lower() == 'exit':
            print("Terminating the program...")
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

if __name__ == "__main__":
    main()
