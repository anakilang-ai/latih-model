import logging
from utils import logging_config, BartGenerator
from trainprocess import path

# Configure logging
logging_config('log_model', 'generator_test.log')

def main():
    # Initialize the generator
    generator = BartGenerator(path)

    logging.info("Generator initialized successfully.")

    while True:
        question = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ").strip()

        if question.lower() == 'exit':
            print("Terminating the program...")
            logging.info("Program terminated by user.")
            break

        try:
            # Generate an answer
            answer = generator.generate_answer(question)
            print(f"Jawaban: {answer}")

            # Log the result
            logging.info(f"Model Path: {generator.model_path}")
            logging.info(f"Pertanyaan: {question}")
            logging.info(f"Jawaban: {answer}")
            logging.info("------------------------------------------\n")
        except Exception as e:
            # Log the error with a detailed message
            print(f"Error: {e}")
            logging.error(f"An error occurred while generating the answer. Details: {e}")

if __name__ == "__main__":
    main()
