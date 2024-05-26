import hashlib
import tkinter as tk
from tkinter import messagebox

def hash_sifre(sifre):
    hash_obj = hashlib.sha256()
    hash_obj.update(sifre.encode('utf-8'))
    return hash_obj.hexdigest()

def kaydet():
    kullanici_adi = kullanici_adi_entry.get()
    sifre = sifre_entry.get()

    if not kullanici_adi or not sifre:
        messagebox.showwarning("Uyarı", "Kullanıcı adı ve şifre boş bırakılamaz.")
        return

    hashli_sifre = hash_sifre(sifre)

    with open("kullanicilar.txt", "a") as dosya:
        dosya.write(f"{kullanici_adi},{hashli_sifre}\n")

    messagebox.showinfo("Başarılı", "Kayıt işlemi başarıyla tamamlandı.")

root = tk.Tk()
root.title("Kayıt Ekranı")

tk.Label(root, text="Kullanıcı Adı:").pack(pady=10)
kullanici_adi_entry = tk.Entry(root)
kullanici_adi_entry.pack(pady=10)

tk.Label(root, text="Şifre:").pack(pady=10)
sifre_entry = tk.Entry(root, show="*")
sifre_entry.pack(pady=10)

kayit_button = tk.Button(root, text="Kaydol", command=kaydet)
kayit_button.pack(pady=20)

root.mainloop()
import hashlib
import tkinter as tk
from tkinter import messagebox

def hash_sifre(sifre):
    hash_obj = hashlib.sha256()
    hash_obj.update(sifre.encode('utf-8'))
    return hash_obj.hexdigest()

def kontrol_et():
    kullanici_adi = kullanici_adi_giris.get()

    if not kullanici_adi:
        messagebox.showwarning("Uyarı", "Kullanıcı adı boş bırakılamaz.")
        return

    with open("kullanicilar.txt", "r") as dosya:
        for satir in dosya:
            kayit_adi, hashli_sifre = satir.strip().split(',')
            if kullanici_adi == kayit_adi:
                muhtemel_parolalar_ile_karsilastir(hashli_sifre)
                return

    messagebox.showerror("Hata", "Kullanıcı adı bulunamadı.")

def muhtemel_parolalar_ile_karsilastir(hashli_sifre):
    with open("muhtemelsifre.txt", "r") as dosya:
        for muhtemel_sifre in dosya:
            muhtemel_sifre = muhtemel_sifre.strip()
            if hash_sifre(muhtemel_sifre) == hashli_sifre:
                messagebox.showinfo("Başarılı", f"Eşleşme bulundu! Muhtemel Parola: {muhtemel_sifre}")
                return

    messagebox.showinfo("Sonuç", "Eşleşme bulunamadı.")

root = tk.Tk()
root.title("Parola Kontrol Ekranı")

tk.Label(root, text="Kullanıcı Adı:").pack(pady=10)
kullanici_adi_giris = tk.Entry(root)
kullanici_adi_giris.pack(pady=10)

kontrol_button = tk.Button(root, text="Parola Kontrol Et", command=kontrol_et)
kontrol_button.pack(pady=20)

root.mainloop()