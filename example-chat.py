import random
import logging

# Configure logging
logging.basicConfig(
    filename='chat_game.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# List of random responses
random_answers = [
    "Saya baik-baik saja, terima kasih!",
    "Hari saya sangat menyenangkan!",
    "dawdawdawd!",
    "abccsadwadwa",
    "Saya baik-baik saja, terima kasih!",
    "Hari saya sangat menyenangkan!",
]

def main():
    print("Selamat datang di Game Chat! Ketik 'keluar' untuk mengakhiri obrolan.")
    logging.info("Chat game started.")

    while True:
        # Get user input
        user_question = input("Anda: ").strip()

        # Check if user wants to exit
        if user_question.lower() == 'keluar':
            print("Bot: Terima kasih sudah mengobrol! Sampai jumpa!")
            logging.info("User exited the chat.")
            break

        # Generate a random response
        response = get_random_response(random_answers)
        print(f"Bot: {response}")

        # Log the interaction
        logging.info(f"User: {user_question}")
        logging.info(f"Bot: {response}")

def get_random_response(answers_list):
    """Return a random response from the provided list."""
    return random.choice(answers_list)

if __name__ == "__main__":
    main()
