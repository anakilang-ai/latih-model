import logging
from utils import logging_config, BartGenerator
from trainprocess import path

logging_config('log_model', 'generator_test.log')

def main():
    generator = BartGenerator(path)

    while True:
        question = input("Masukkan pertanyaan (atau ketik 'keluar' untuk keluar): ").strip()

        if question.lower() == 'keluar':
            print("Menghentikan program...")
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

if _name_ == "_main_":
    main()