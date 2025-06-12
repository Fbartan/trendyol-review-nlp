from gensim.models import CoherenceModel
import pandas as pd
from gensim import corpora

if __name__ == "__main__":
    # Veriyi oku
    df = pd.read_excel(r"C:\Users\FK\Desktop\yayın için çalışma\trendyol.xlsx")

    # Yorumları temizleyip token'lara ayıralım
    texts = df['Yorum Metni'].dropna().astype(str).apply(lambda x: x.split())

    # BERTopic konuları
    bertopic_topics = [
        ['37', '38', '39', '36', 'giyiyorum', '40', 'normalde', '42', '44', 'giyiyordum'],
        ['pantolon', 'pantalon', 'pantolonu', 'pantolonun', 'pantolonlar', 'pantolona', 'pantolonlardan', 'pantolonları', 'pantolonum', 'pantolonla'],
        ['gözlük', 'gözlükler', 'gözlüğü', 'gözlüğüm', 'yüz', 'bilge', 'gözlüğün', 'yüzüme', 'tipine', 'yüze'],
        ['37', '39', '38', 'ayağım', '36', 'giyiyorum', 'normalde', '40', 'ayaklarım', 'ayağıma'],
        ['bedenimi', 'bedenimden', 'beden', 'bedenime', 'almama', 'bedenini', 'bedenim', 'buyuk', 'bedeni', 'büyük'],
        ['kiloyum', 'kilo', 'boy', '60', '50', 'kg', 'kilom', '55', 'boyum', '62'],
        ['hediyeniz', 'hediyeler', 'içinde', 'hediye', 'hediyeleriniz', 'teşekkür', 'minik', 'ayrıca', 'yanındaki', 'icinde'],
        ['renk', 'rengi', 'koyu', 'renkler', 'renkle', 'farklı', 'soluk', 'attı', 'renkte', 'renkleri'],
        ['saç', 'saçlarım', 'saçı', 'gür', 'saçım', 'saçımı', 'saçlarımı', 'telli', 'saçları', 'tutuyor'],
        ['yıldız', 'yıldızı', 'beş', 'çıksın', 'görünsün', '10', 'verirdim', 'üstte', 'gözüksün', 'öne']
    ]

    # Sözlük oluştur
    dictionary = corpora.Dictionary(texts)

    # Coherence hesapla (multiprocessing kapalı olacak!)
    coherence_model_bertopic = CoherenceModel(
        topics=bertopic_topics, 
        texts=texts, 
        dictionary=dictionary, 
        coherence='c_v',   # Burada C_V seçtik çünkü npmi multiprocessing kullanıyor
        processes=1        # BU ÇOK ÖNEMLİ -> multiprocessing OFF
    )
    coherence_score_bertopic = coherence_model_bertopic.get_coherence()

    print(f"🎯 BERTopic C_V Coherence Score: {coherence_score_bertopic}")
