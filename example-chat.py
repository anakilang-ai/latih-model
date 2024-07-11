import random

# Daftar jawaban acak
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
    "Buku favorit saya adalah 'Harry Potter'.",
    "Halo! Apa kabar?",
    "Apa yang kamu lakukan hari ini?",
    "Pernah nonton film baru-baru ini?",
    "Apa makanan favoritmu?",
    "Ceritakan tentang hobi kamu!",
    "Apa rencana kamu akhir pekan ini?",
    "Apakah kamu suka bermain game?",
    "Sudah makan siang?",
    "Kamu suka olahraga apa?",
    "Apa buku favoritmu?"
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

if __name__ == "__main__":
    main()
