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
            response = answer_generator.get_answer(query)
            print(f"Answer: {response}")

            # Log the details
            logging.info(f"Model Path: {answer_generator.model_path}")
            logging.info(f"Question: {query}")
            logging.info(f"Answer: {response}")
            logging.info("------------------------------------------\n")
        except ValueError as error:
            print(error)

if __name__ == "__main__":
    run()