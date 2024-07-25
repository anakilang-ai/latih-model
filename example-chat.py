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

def chat_bot():
    print("Selamat datang di Obrolan Bot! Ketik 'keluar' untuk mengakhiri percakapan.")
    
    while True:
        # Input pertanyaan dari pengguna
        user_input = input("Anda: ")
        
        # Cek jika pengguna ingin keluar
        if user_input.strip().lower() == 'keluar':
            print("Bot: Terima kasih sudah mengobrol! Sampai jumpa lagi!")
            break
        
        # Pilih jawaban secara acak
        bot_response = random.choice(responses)
        print("Bot: " + bot_response)

if __name__ == "__main__":
    chat_bot()
