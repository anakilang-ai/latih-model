import random

# Daftar pesan obrolan acak
chat_responses = [
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
        # Pilih pesan obrolan acak
        response = random.choice(chat_responses)
        print("Bot: " + response)
        
        # Minta pengguna untuk memasukkan pesan balasan
        user_input = input("Anda: ")
        
        # Periksa apakah pengguna ingin keluar
        if user_input.lower() == 'keluar':
            print("Bot: Terima kasih sudah mengobrol! Sampai jumpa!")
            break

if __name__ == "__main__":
    main()