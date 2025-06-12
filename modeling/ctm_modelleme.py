import pandas as pd
import nltk
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.evaluation.measures import CoherenceNPMI

print("🚀 Kod çalışmaya başladı.")

def main():
    print("📥 Excel verisi yükleniyor...")
    df = pd.read_excel(r"C:\Users\FK\Desktop\yayın için çalışma\trendyol.xlsx", nrows=135000)

    print(f"✅ Excelden okunan yorum sayısı: {len(df)}")
    texts = df["Yorum Metni"].dropna().astype(str).tolist()

    print("🛠️ NLTK kütüphanesi hazır mı diye kontrol...")
    nltk.download("stopwords")

    print("🧹 Temizlik başlıyor...")
    clean_texts = [text for text in texts if isinstance(text, str) and len(text.split()) > 1]

    print(f"✅ Temizlenmiş yorum sayısı: {len(clean_texts)}")
    print("🔍 İlk 5 yorumu gösteriyorum:")
    for i, text in enumerate(clean_texts[:5], 1):
        print(f"{i}: {text}")

    if not clean_texts:
        print("❌ Uyarı: Hiç temizlenmiş yorum kalmadı, işlemi durduruyoruz.")
        return

    print("🔤 Vectorizer hazırlanıyor...")
    qt = TopicModelDataPreparation("paraphrase-MiniLM-L6-v2")  # 🔥 384 boyutlu model ismi

    training_dataset = qt.fit(text_for_bow=clean_texts, text_for_contextual=clean_texts)

    print("🧠 CTM Modeli oluşturuluyor...")
    ctm_model = CombinedTM(bow_size=len(qt.vocab),
                           contextual_size=384,    # 🔥 384 olması lazım
                           n_components=100,
                           num_epochs=5,
                           batch_size=64)

    print("🏋️‍♂️ Model eğitiliyor, sabırla bekleyin...")
    ctm_model.fit(training_dataset)

    print("📚 En iyi kelimeler toplanıyor...")
    topics = ctm_model.get_topic_lists(10)
    for idx, topic in enumerate(topics):
        print(f"Konu {idx}: {topic}")

    print("🧪 NPMI Coherence hesabı yapılıyor...")
    tokenized_texts = [text.split() for text in clean_texts]
    coherence = CoherenceNPMI(texts=tokenized_texts, topics=ctm_model.get_topic_lists(10))
    print(f"🎯 NPMI Coherence Score: {coherence.score()}")

    print("💾 Model kaydediliyor...")
    ctm_model.save(models_dir="ctm_model_trendyol")
    print("✅ Model başarıyla kaydedildi.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
