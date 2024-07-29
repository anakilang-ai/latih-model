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

def main():
    print("Selamat datang di Game Chat! Ketik 'keluar' untuk mengakhiri obrolan.")
    
    while True:
        # Minta pengguna untuk memasukkan pertanyaan
        user_question = input("Anda: ")
        
        # Periksa apakah pengguna ingin keluar
        if user_question.lower() == 'keluar':
            print("Bot: Terima kasih sudah mengobrol! Sampai jumpa!")
            break
        
        # Pilih jawaban acak
        response = random.choice(random_answers)
        print("Bot: " + response)

if name == "main":
    main()