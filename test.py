import logging
from utils import logging_config, BartGenerator
from trainprocess import path

# Configure logging
logging_config('log_model', 'generator_test.log')

def main():
    generator = BartGenerator(path)

    while True:
        question = input("Masukkan pertanyaan (atau ketik 'keluar' untuk keluar): ").strip()

        if question.lower() == 'keluar':
            print("Mengakhiri program...")
            break
        
        try:
            answer = generator.generate_answer(question)
            print(f"Jawaban: {answer}")

            # Log the result
            logging.info(f"Model Path: {generator.model_path}")
            logging.info(f"Pertanyaan: {question}")
            logging.info(f"Jawaban: {answer}")
            logging.info("------------------------------------------")
        except ValueError as error:
            print(f"Error: {error}")

if __name__ == "__main__":
    main()