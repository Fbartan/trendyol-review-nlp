import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

turkish_stopwords = [
    "bir", "ve", "bu", "da", "ne", "için", "ile", "gibi", "çok", "ama",
    "de", "mi", "mı", "mu", "mü", "ben", "sen", "o", "biz", "siz", "onlar", "şu",
    "ki", "daha", "en", "her", "kadar", "ise", "diye", "ya", "ya da", "hep", "hiç"
]

# Excel dosyasını oku
excel_yolu = r"C:\Users\FK\Desktop\yayın için çalışma\yorumlar.xlsx"
df = pd.read_excel(excel_yolu)

# Yorum metinleri
yorumlar = df["Yorum Metni"].dropna().astype(str).tolist()

vectorizer_model = CountVectorizer(stop_words=turkish_stopwords)

# BERTopic modelini oluştur
model = BERTopic(language="turkish", vectorizer_model=vectorizer_model)

# Modeli eğit
print("⏳ Model eğitiliyor...")
konular, olasiliklar = model.fit_transform(yorumlar)
print("✅ Model eğitimi bitti")

# İlk 10 konuyu göster
print("\n📌 En çok geçen 10 konu:")
for topic_id in model.get_topics().keys():
    if topic_id != -1:
        print(f"Konu {topic_id}: {model.get_topic(topic_id)}")
        if topic_id >= 9:
            break

model.visualize_topics().show()

# Konu dağılımını çubuk grafikte göster
model.visualize_barchart(top_n_topics=10).show()

print("\n🔍 Konu 0'a ait örnek yorumlar:")
for yorum in model.get_representative_docs(0):
    print("-", yorum)

print("💾 Konular Excel dosyasına ekleniyor...")
df["Konu"] = konular
df.to_excel("yorumlar_konularla.xlsx", index=False)
print("✅ Kaydedildi: yorumlar_konularla.xlsx")

# 🔠 Her konu için kelime bulutu oluştur
output_dir = "kelime_bulutlari"
os.makedirs(output_dir, exist_ok=True)

print("🌥 Kelime bulutları oluşturuluyor...")
for topic_id in model.get_topics().keys():
    if topic_id == -1:
        continue
    topic_words = model.get_topic(topic_id)
    if not topic_words:
        continue
    word_freq = {word: score for word, score in topic_words}
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Konu {topic_id} Kelime Bulutu", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/konu_{topic_id}_bulut.png")
    plt.close()

print(f"✅ Tüm kelime bulutları '{output_dir}' klasörüne kaydedildi.")

# 🔠 Tüm yorumlardan oluşan genel kelime bulutu oluştur
print("🌐 Tüm yorumlar için genel kelime bulutu oluşturuluyor...")
all_text = " ".join(yorumlar)
wc_all = WordCloud(width=1000, height=500, background_color='white', stopwords=turkish_stopwords).generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wc_all, interpolation='bilinear')
plt.axis("off")
plt.title("Genel Kelime Bulutu", fontsize=18)
plt.tight_layout()
plt.savefig("kelime_bulutlari/genel_kelime_bulutu.png")
plt.show()
print("✅ Genel kelime bulutu oluşturuldu ve kaydedildi.")
