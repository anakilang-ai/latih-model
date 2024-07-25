import random

# Daftar jawaban acak
responses = [
    "Saya baik-baik saja, terima kasih!",
    "Hari ini sangat menyenangkan!",
    "Saya baru saja selesai menonton film yang menarik.",
    "Saya sangat suka pizza, itu adalah makanan favorit saya.",
    "Saya memiliki hobi bermain gitar.",
    "Akhir pekan ini saya berencana untuk hiking.",
    "Ya, saya suka bermain game, terutama yang bergenre strategi.",
    "Belum, saya belum makan siang.",
    "Saya menikmati bermain sepak bola.",
    "Buku favorit saya adalah 'Harry Potter'."
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
