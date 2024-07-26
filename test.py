import logging
from utils import QADataset, logging_config, BartGenerator
from utilstest import logging_config, BartGenerator
from trainprocess import path


logging_config('log_model', 'generator_test.log')

def main():
    generator = BartGenerator(model_path='model/bart_coba{num}-{epoch}-{batch_size}')

    while True:
        question = input("Masukkan pertanyaan (atau ketik 'exit' untuk keluar): ").strip()

        if question.lower() == 'exit':
            print("Terminating the program...")
            break

        try:
            answer = generator.generate_answer(question, max_length=100)
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
