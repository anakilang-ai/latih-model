import random

# List of random answers
random_answers = [
    "Saya baik-baik saja, terima kasih!",
    "Hari saya sangat menyenangkan!",
    "Saya baru saja menonton film yang bagus.",
    "Saya suka pizza, itu makanan favorit saya.",
    "Saya hobi bermain gitar.",
    "Akhir pekan ini saya berencana pergi hiking.",
    "Ya, saya suka bermain game, terutama game strategi.",
    "Belum, saya belum makan siang.",
    "Saya suka bermain sepak bola.",
    "Buku favorit saya adalah 'Harry Potter'."
]

def main():
    print("Selamat datang di Game Chat! Ketik 'keluar' untuk mengakhiri obrolan.")
    
    while True:
        # Ask the user for input
        user_question = input("Anda: ")
        
        # Check if the user wants to exit
        if user_question.lower() == 'keluar':
            print("Bot: Terima kasih sudah mengobrol! Sampai jumpa!")
            break
        
        # Select a random response
        response = random.choice(random_answers)
        print("Bot: " + response)

if __name__ == "__main__":
    main()
