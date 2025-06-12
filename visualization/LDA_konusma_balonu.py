import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")
nltk.download("stopwords")

# 🔄 Excel'den yorumları oku
df = pd.read_excel(r"C:\Users\FK\Desktop\yayın için çalışma\trendyol.xlsx")

yorumlar = df["Yorum Metni"].dropna().astype(str).tolist()

# 🔠 Türkçe stopword listesi
stop_words = set(stopwords.words("turkish"))

# 🧹 Ön işleme (tokenizasyon ve stopword temizliği)
texts = [
    [kelime for kelime in word_tokenize(cumle.lower()) if kelime.isalpha() and kelime not in stop_words]
    for cumle in yorumlar
]

# 📚 Gensim için sözlük ve gövde oluştur
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# 🧠 LDA modelini eğit
lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)

# 🧠 Konulardan kelime ve skorlarını al
kelime_skorlari = {}
for topic_id in range(lda_model.num_topics):
    topic = lda_model.show_topic(topic_id, topn=20)
    for kelime, skor in topic:
        kelime_skorlari[kelime] = kelime_skorlari.get(kelime, 0) + skor

# 🖼️ Şekil maskesini yükle
mask = np.array(Image.open(r"C:\Users\FK\Desktop\yayın için çalışma\konusmabalonu.png"))

# ☁️ Kelime bulutunu oluştur
wc = WordCloud(
    width=800,
    height=800,
    background_color="white",
    mask=mask,
    contour_color="black",
    contour_width=1,
    font_path=None
).generate_from_frequencies(kelime_skorlari)

# 💾 Kaydet ve göster
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("En Anlamlı LDA Kelimeleri (Şekilli)", fontsize=16)
plt.tight_layout()
plt.savefig("lda_kelime_bulutu_sekilli.png")
plt.show()
# 📊 LDA konularını çubuk grafikle göster (ilk 5 konuyu örnek olarak)
print("📊 En Anlamlı LDA Konuları (İlk 5)")
for topic_id in range(5):  # daha fazlası istenirse aralığı artırabilirsin
    topic_terms = lda_model.show_topic(topic_id, topn=10)
    kelimeler, skorlar = zip(*topic_terms)

    plt.figure(figsize=(8, 4))
    plt.barh(kelimeler, skorlar, color="skyblue")
    plt.xlabel("Ağırlık")
    plt.title(f"Konu {topic_id} - En Anlamlı 10 Kelime")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"lda_konu_{topic_id}_grafik.png")
    plt.show()

