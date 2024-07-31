import random

# Daftar jawaban berdasarkan kategori
answers = {
    "greeting": [
        "Halo! Apa kabar?",
        "Hai, bagaimana hari Anda?",
        "Selamat datang! Ada yang bisa saya bantu?"
    ],
    "farewell": [
        "Terima kasih telah berbicara dengan saya. Sampai jumpa!",
        "Selamat tinggal! Semoga hari Anda menyenangkan.",
        "Sampai jumpa lagi! Terima kasih sudah mengobrol."
    ],
    "default": [
        "Maaf, saya tidak mengerti pertanyaan Anda.",
        "Bisa jelaskan lebih lanjut?",
        "Saya tidak yakin tentang itu. Coba tanya yang lain."
    ]
}

def chat():
    print("Selamat datang di Game Chat! Ketik 'keluar' untuk mengakhiri obrolan.")
    
    exit_keywords = ['keluar', 'exit', 'quit', 'stop']
    
    while True:
        # Minta pengguna untuk memasukkan pertanyaan
        user_question = input("Anda: ")
        
        # Periksa apakah pengguna ingin keluar
        if any(keyword in user_question.lower() for keyword in exit_keywords):
            response = random.choice(answers["farewell"])
            print("Bot: " + response)
            break
        
        # Tentukan jawaban berdasarkan pola pertanyaan
        if 'halo' in user_question.lower() or 'hai' in user_question.lower():
            response = random.choice(answers["greeting"])
        else:
            response = random.choice(answers["default"])
        
        print("Bot: " + response)

if __name__ == "__main__":
    chat()
