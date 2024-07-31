import random

# List of random responses
responses = [
    "Saya baik-baik saja, terima kasih!",
    "Hari saya sangat menyenangkan!",
    "dawdawdawd!",
    "abccsadwadwa",
    "Hari saya sangat menyenangkan!"
] * 20  # Repeat to maintain similar length

def chat():
    print("Selamat datang di Chat Bot! Ketik 'exit' untuk mengakhiri obrolan.")
    
    while True:
        # Get user input
        user_input = input("Kamu: ")
        
        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Bot: Terima kasih sudah berbicara! Sampai jumpa!")
            break
        
        # Select a random response
        bot_response = random.choice(responses)
        print("Bot: " + bot_response)

if __name__ == "__main__":
    chat()