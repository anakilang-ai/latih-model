import logging
from utils import logging_config, BartGenerator
from trainprocess import path

# Configure logging
logging_config('log_model', 'generator_test.log')

def main():
    # Initialize the generator
    try:
        generator = BartGenerator(path)
        logging.info("Generator initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize the generator: {e}")
        print(f"Error initializing the generator: {e}")
        return

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
            logging.info("Response successfully generated and logged.")
        except Exception as e:
            # Log the error with a detailed message
            logging.error(f"An error occurred while generating the answer. Details: {e}")
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
