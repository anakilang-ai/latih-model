import random

# Daftar jawaban acak
random_answers = [
    "Saya baik-baik saja, terima kasih!",
    "Hari saya sangat menyenangkan!",
    "dawdawdawd!",
    "Hari saya sangat menyenangkan!",
    "abccsadwadwa",
    "Hari saya sangat menyenangkan!",
        "Hari saya sangat menyenangkan!",
    "abccsadwadwa",
        "Saya baik-baik saja, terima kasih!",
    "Hari saya sangat menyenangkan!",
    "dawdawdawd!",
    "Hari saya sangat menyenangkan!",
    "abccsadwadwa",
    "Hari saya sangat menyenangkan!",
        "Hari saya sangat menyenangkan!",
    "abccsadwadwa",
    "Hari saya sangat menyenangkan!"
]

def main():
    print("Selamat datang di Game Chat! Ketik 'keluar' untuk mengakhiri obrolan.")
    print("kiww")
    
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