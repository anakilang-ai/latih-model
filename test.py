import logging
from utils import setup_logging, BartAnswerGenerator, get_answer
from trainprocess import model_path

setup_logging('log_model', 'generator_test.log')

def run():
    answer_generator = BartAnswerGenerator(model_path)

    while True:
        query = input("Enter your question (or type 'exit' to quit): ").strip()

        if query.lower() == 'exit':
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

if _name_ == "_main_":
    main()