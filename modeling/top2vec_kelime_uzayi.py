from top2vec import Top2Vec
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 🔄 1. Modeli yükle
print("📂 Model yükleniyor...")
model = Top2Vec.load("top2vec_model")

# 🔍 2. Konu vektörlerini al
topic_vectors = model.topic_vectors

# 🧭 3. TSNE ile 2D uzaya indir
print("🧠 Vektörler indirgeniyor...")
tsne = TSNE(n_components=2, perplexity=15, max_iter=1000, random_state=42)
topic_embeddings_2d = tsne.fit_transform(topic_vectors)

# 📊 4. Görselleştir
print("🎯 Konu vektörleri görselleştiriliyor...")
plt.figure(figsize=(10, 6))
plt.scatter(topic_embeddings_2d[:, 0], topic_embeddings_2d[:, 1], c='blue', s=50)
plt.title("Top2Vec Konu Vektörleri (TSNE ile)")
plt.xlabel("Boyut 1")
plt.ylabel("Boyut 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("top2vec_konu_uzayi.png")
plt.show()

print("✅ Görsel oluşturuldu: top2vec_konu_uzayi.png")
