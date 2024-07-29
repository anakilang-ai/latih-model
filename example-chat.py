import random

# List of random responses
responses = [
    "I'm doing well, thank you!",
    "My day has been fantastic!",
    "I just watched a great movie.",
    "I love pizza, it's my favorite food.",
    "I enjoy playing the guitar.",
    "This weekend I plan to go hiking.",
    "Yes, I like playing games, especially strategy games.",
    "No, I haven't had lunch yet.",
    "I like playing soccer.",
    "My favorite book is 'Harry Potter'."
]

def chat_game():
    print("Welcome to the Chat Game! Type 'exit' to end the chat.")
    
    while True:
        # Ask the user for input
        user_input = input("You: ")
        
        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Bot: Thanks for chatting! See you later!")
            break
        
        # Select a random response
        bot_response = random.choice(responses)
        print("Bot: " + bot_response)

if __name__ == "__main__":
    chat_game()