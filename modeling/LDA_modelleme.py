import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import nltk

nltk.download("punkt")
nltk.download("stopwords")

# 📥 Yorumları oku
df = pd.read_excel(r"C:\Users\FK\Desktop\yayın için çalışma\trendyol.xlsx")

documents = df["Yorum Metni"].dropna().astype(str).tolist()

# 🧹 Temizleme ve tokenizasyon
stop_words = set(stopwords.words("turkish"))
tokenized_docs = []
for doc in documents:
    tokens = word_tokenize(doc.lower())
    clean_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    tokenized_docs.append(clean_tokens)

# 📚 LDA için sözlük ve doküman frekansları
dictionary = corpora.Dictionary(tokenized_docs)
corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

# 🧠 LDA modelini eğit
lda_model = models.LdaModel(corpus=corpus,
                            id2word=dictionary,
                            num_topics=10,
                            random_state=42,
                            passes=10)

print("📌 LDA Modeli - Konu Temsilci Kelimeleri:\n")
for i in range(10):  # Konu sayını değiştir
    topic = lda_model.show_topic(i, topn=10)  # ilk 10 kelimeyi al
    kelimeler = [kelime for kelime, skor in topic]
    print(f"Konu {i}: {kelimeler}")

# 🌥 Her konu için kelime bulutu
for topic_id in range(10):
    plt.figure(figsize=(10, 5))
    topic_words = dict(lda_model.show_topic(topic_id, topn=30))
    wc = WordCloud(width=1000, height=500, background_color='white').generate_from_frequencies(topic_words)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"Konu {topic_id} - Kelime Bulutu", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"lda_konu_{topic_id}_bulut.png")
    plt.show()
