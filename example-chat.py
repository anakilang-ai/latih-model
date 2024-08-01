import random

# List jawaban acak
responses_list = [
    "Saya baik-baik saja, terima kasih!",
    "Hari saya sangat menyenangkan!",
    "dawdawdawd!",
    "abccsadwadwa",
    "Hari saya sangat menyenangkan!",
    # ... (tambahkan semua elemen lain yang ada di daftar)
]

def start_chat():
    print("Selamat datang di Game Chat! Ketik 'keluar' untuk mengakhiri obrolan.")
    
    while True:
        # Ambil pertanyaan pengguna
        user_input = input("Anda: ")
        
        # Periksa apakah pengguna ingin keluar
        if user_input.lower() == 'keluar':
            print("Bot: Terima kasih sudah mengobrol! Sampai jumpa!")
            break
        
        # Pilih jawaban acak dari daftar
        bot_response = random.choice(responses_list)
        print("Bot: " + bot_response)

if __name__ == "__main__":
    start_chat()