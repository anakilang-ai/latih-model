import random

# Daftar jawaban berdasarkan kategori
answers = {
    "baik": [
        "Saya baik-baik saja, terima kasih!",
        "Hari saya sangat menyenangkan!",
    ],
    "biasa": [
        "dawdawdawd!",
        "abccsadwadwa",
    ],
    "lelah": [
        "Hari saya sangat menyenangkan!",
        "Saya baik-baik saja, terima kasih!",
    ],
}

def chat():
    print("Selamat datang di Game Chat! Ketik 'keluar' untuk mengakhiri obrolan.")
    
    exit_keywords = ['keluar', 'exit', 'quit', 'stop']
    
    while True:
        # Minta pengguna untuk memasukkan pertanyaan
        user_question = input("Anda: ")
        
        # Periksa apakah pengguna ingin keluar
        if any(keyword in user_question.lower() for keyword in exit_keywords):
            print("Bot: Terima kasih sudah mengobrol! Sampai jumpa!")
            break
        
        # Pilih jawaban berdasarkan kata kunci dalam pertanyaan
        if 'baik' in user_question.lower():
            response = random.choice(answers['baik'])
        elif 'biasa' in user_question.lower():
            response = random.choice(answers['biasa'])
        elif 'lelah' in user_question.lower():
            response = random.choice(answers['lelah'])
        else:
            response = random.choice(answers['baik'])  # Default response
        
        print("Bot: " + response)

if __name__ == "__main__":
    chat()
