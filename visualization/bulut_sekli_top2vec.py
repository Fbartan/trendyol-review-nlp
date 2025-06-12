from top2vec import Top2Vec
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import defaultdict

# 🔄 Modeli yükle
model = Top2Vec.load("top2vec_model")

# 📌 Konu kelimeleri ve skorlarını al
topics_words, topics_scores, topic_nums = model.get_topics()
top_n = 10
top_topic_indices = topic_nums[:top_n]

# 🧠 Tüm kelimeleri toplayacak sözlük
combined_word_scores = defaultdict(float)
for i in top_topic_indices:
    for word, score in zip(topics_words[i], topics_scores[i]):
        combined_word_scores[word] += score

# 🖼️ Maskeyi yükle
mask_path = "bulut3.png"  # kendi maskeni buraya koy
mask = np.array(Image.open(mask_path))

# 🌥 Kelime bulutu oluştur
wc = WordCloud(width=1200, height=800, background_color='white', mask=mask, contour_color='black').generate_from_frequencies(combined_word_scores)

# 💾 Görseli kaydet
plt.figure(figsize=(14, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("En Anlamlı 10 Konunun Şekilli Kelime Bulutu", fontsize=16)
plt.tight_layout()
plt.savefig("top2vec_kelime_bulutlari/top10_topics_wordcloud_masked.png")
plt.show()
