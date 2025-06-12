from top2vec import Top2Vec
import pandas as pd

# Excel dosyasını oku
excel_path = r"C:\Users\FK\Desktop\yayın için çalışma\trendyol.xlsx"
df = pd.read_excel(excel_path)

# Yorumları liste haline getir
yorumlar = df["Yorum Metni"].dropna().astype(str).tolist()

# Modeli eğit
print("⏳ Top2Vec modeli eğitiliyor...")
model = Top2Vec(documents=yorumlar, speed="learn", workers=4)
print("✅ Model eğitildi.")

# İlk 10 konuyu getir
top_words, word_scores, topic_nums = model.get_topics(10)

# Konuları yazdır
for i, (kelimeler, skorlar) in enumerate(zip(top_words, word_scores)):
    print(f"\n🔹 Konu {i}:")
    for kelime, skor in zip(kelimeler, skorlar):
        print(f"  {kelime} ({skor:.4f})")
model.save("top2vec_model")
print("✅ Model kaydedildi: top2vec_model")
