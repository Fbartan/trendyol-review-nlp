from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
mask = np.array(Image.open(r"C:\Users\FK\Desktop\yayın için çalışma\karekonusmabalonu.png"))

# Tüm konulardaki kelimeleri birleştiriyoruz
topics_words = [
    ['rengi', 'kötü', 'alakası', 'ince', 'değil', 'renk', 'yanlış', 'zinciri', 'yok', 'iade'],
    ['çok', 'memnun', 'kaldım', 'kendime', 'beğendik', 'almıştım', 'cok', 'hediye', 'arkadaşım', 'annem'],
    ['oldu', 'boy', 'kilo', 'kg', 'xl', 'boyu', 'beden', 'kiloyum', 'bol', '60'],
    ['duracak', 'it', 'azıcık', 'delik', 'parmaklar', 'koymuslar', 'bigudi', 'düşünmeyin', 'kalırsanız', 'göre'],
    ['ama', 'bu', 'daha', 'kumaşı', 'bir', 'büyük', 'biraz', 'iade', 'da', 'rağmen'],
    ['gelmedi', 'değil', 'iade', 'kötü', 'alakası', 'gösteriyor', 'yok', 'iç', 'bana', 'kalitesiz'],
 ['kalite', 'mükemmel', 'harika', 'çox', 'almayın', 'mukemmel', 'orta', 'alakası', 'kelimeyle', 'değer'],
 ['eder', 'süper', 'dare', 'idare', 'güzellll', 'güzeller', 'fena', 'güzelll', 'güzell', 'sporda'],
 ['bir', 've', 'da', 'sipariş', 'ayakkabı', 'için', 'ürün', 'aynı', 'bu', 'daha'],
 ['güzelll', 'güzellll', 'güzeller', 'kararsız', 'güzelllll', 'tür', 'parmağım', 'aşağı', 'kısmındaki', 'yazmışlar'],
 ['ürün', 'geldi', 'gibi', 'fakat', 'sadece', 'aynı', 'biraz', 'göre', 'ama', 'ile'],
 ['oldu', 'büyük', '38', 'giyiyorum', 'ama', '37', 'biraz', 'ben', 'numara', 'dar'],
 ['daha', 'rengini', 'kadar', 'baya', 'diğer', 'tane', 'ki', 'var', 'en', 'bu'],
 ['tavsiye', 'almanızı', 'ediyorum', 'ederim', 'gönül', 'ulaştı', 'hızlı', 'rahatlığıyla', 'numaranızı', 'kargo'],
 ['çok', 'için', 'de', 've', 'ayrıca', 'gerçekten', 'uygun', 'teşekkürler', 'da', 'teşekkür'],
 ['yok', 'bu', 'kötü', 'var', 'diye', 'ne', 'fakat', 'kadar', 'asla', 'rengi'],
 ['ama', 'daha', 'da', 'de', 'rengi', 'bu', 'biraz', 'var', 'iade', 'diye'],
 ['büyük', 'numara', 'ama', 'küçük', 'bir', 'oldu', 'biraz', '38', 'beden', 'ben'],
 ['begendim', 'beğendim', 'begendik', 'edecek', 'değişiyor', 'belirtmek', 'şansım', 'çözüm', '1000', 'yapıda'],
 ['boy', 'boyu', 'oldu', 'kilo', 'beden', 'bol', 'kg', '60', 'pantolon', 'kiloyum'],
 ['kullanıyorum', 'severek', 'kelime', 'her', 'tek', 'kelimeyle', 'harika', 'bayıldım', 'muhteşem', 'rengine'],
 ['kullanıyorum', 'severek', 'bayıldım', 'muhteşem', 'herkes', 'çanta', 'kelimeyle', 'kombine', 'canta', 'çantaya'],
 ['yok', 'ile', 'kadar', 'bile', 'böyle', 'daha', 'değil', 'hiç', 'bu', 'kesinlikle'],
 ['paketleme', 'hızlı', 'ayrıca', 've', 'teşekkür', 'teşekkürler', 'gerçekten', 'ediyorum', 'kargo', 'satıcıya'],
 ['mükemmel', 'harika', 'tek', 'kelimeyle', 'bayıldım', 'aldırın', 'kelime', 'çox', 'kesinlikle', 'muhteşem'],
 ['çok', 've', 'da', 'de', 'ben', 'bir', 'ayakkabı', 'için', 'bu', 'daha'],
 ['sadece', 'rengi', 'fakat', 'ürün', 'yok', 'var', 'biraz', 'ne', 'kadar', 'diye'],
 ['şık', 'tatlı', 'zarif', 'duruyor', 'sevmiyorum', 'tip', 'olmamıştı', 'sevimli', 'üretim', 'duracak'],
 ['fiyat', 'performans', 'ürünü', 'bedeninizi', 'numaranizi', 'alabilirsiniz', 'kendi', 'alin', 'rahatlığıyla', 'gönül'],
 ['güzel', 'guzel', 'durumu', 'anlamış', 'santim', 'olmazdı', 'düşünce', 'yukarıda', 'şansım', 'iadem'],
 ['alın', 'alınmalı', 'bedeninizi', 'tam', 'alınabilir', 'alin', 'yazlık', 'beden', 'kalıp', 'bedeninizden'],
 ['her', 'önce', 'iki', 'almıştım', 'hem', 'rengini', 'daha', 'ki', 'bu', 'kendime'],
 ['ürün', 'hızlı', 'geldi', 've', 'kargo', 'ulaştı', 'gayet', 'göre', 'güzel', 'elime'],
 ['begendim', 'beğendim', 'güzel', 'guzel', 'sevmiyorum', 'kalınca', 'parmağım', 'sıkma', 'değilseniz', 'vurabilir'],
 ['alacaksanız', 'aldigim', 'çinde', 'bırakıp', 'olmamıştı', 'sevmiyorum', 'şansım', 'verilebilir', 'merak', 'bulamadığım'],
 ['göründüğü', 'beklediğim', 'gibi', 'geldi', 'görseldeki', 'istediğim', 'fotoğraftaki', 'görüldüğü', 'gözüktüğü', 'beklediğimden'],
 ['çanta', 'hem', 'kadar', 'da', 've', 'kaliteli', 'de', 'bu', 'fiyata', 'çok'],
 ['alınmalı', 'alın', 'alınabilir', 'alin', 'tam', 'bedeninizi', 'beden', 'buyuk', 'kalıp', 'bedeninizden'],
 ['güzel', 'ötürü', 'alakalı', 'guzel', 'alayım', 'yazmışlar', 'gayet', 'sevmiyorum', 'gözüme', 'dört'],
 ['hızlı', 'sorunsuz', 'elime', 'sağlam', 'ulaştı', 'kargo', 'teşekkürler', 'şekilde', 'teşekkür', 'ürünler'],
 ['oldu', 'beden', '36', 'kiloyum', 'giyiyorum', 'boy', '38', '37', 'küçük', 'kilo'],
 ['güzel', 'guzel', 'duracak', 'alakalı', 'olmamıştı', 'begendim', 'korkmuştum', 'problemi', 'birisiyim', 'kalıplarda'],
 ['boy', 'kilo', 'oldu', 'kg', 'boyu', 'kiloyum', 'uzun', 'dar', 'beden', 'boyum'],
 ['aldım', 'ama', 'ben', 'oldu', 'beden', 'bir', 'küçük', 'pantolon', 'uzun', 'daha'],
 ['fiyatına', 'göre', 'gayet', 'gore', 'fiyatina', 'güzel', 'ürün', 'kullanışlı', 'arkadaki', 'kaliteli'],
 ['güzel', 'tabanlık', 'hani', 'değişiyor', 'guzel', 'ordan', 'benimki', 'demiş', 'giymeyi', 'sıkma'],
 ['beğendi', 'anneme', 'kızıma', 'begendi', 'hediye', 'annem', 'arkadaşıma', 'aldık', 'beğenildi', 'ablama'],
 ['ederim', 'teşekkür', 'tavsiye', 'herkese', 'ederiz', 'ediyorum', 'ayrıca', 'hediyeniz', 'paketleme', 'satıcıya'],
 ['beden', 'kilo', 'kg', 'boy', 'alınmalı', '58', 'xl', '65', '50', '60'],
 ['tavsiye', 'ederim', 'herkese', 'ederiz', 'hediyeniz', 'teşekkürler', 'teşekkür', 'sağlam', 'özenli', 'tşk'],
 ['tabanlık', 'alakalı', 'giymeyi', 'düşünce', 'parmağım', 'ötürü', 'sini', 'dedikleri', 'yandan', 'parmaklarımı'],
 ['çox', 'bəyəndim', 'gəldi', 'kelimeyle', 'bayılarak', 'bayildimm', 'cox', 'mükemmel', 'aylardır', 'bayıldım'],
 ['çok', 'duruyor', 'ben', 'güzel', 'duruşu', 'gerçekten', 'fiyata', 'hoş', 've', 'kaliteli'],
 ['ayakkabi', 'duruyor', 'rahat', 'şık', 'ayakkabı', 'duruyo', 'güzel', 've', 'fiyatına', 'ayakta'],
 ['cox', 'süper', 'mukemmel', 'mükemmel', 'şort', 'super', 'çox', 'sporda', 'harikaaa', 'yerine'],
 ['begendim', 'beğendim', 'guzel', 'güzel', 'yukarıda', 'firmayı', 'olmuyordu', 'anlamış', 'denediğim', 'sevmiyorum'],
 ['fiyat', 'performans', 'bedenizi', 'istedigim', 'ürünü', 'istediğim', 'sapları', 'alin', 'jest', 'allahtan'],
 ['oldu', 'bir', 'büyük', 'aldım', 'tam', 'giyiyorum', 'da', 'ben', 'ama', 'beden'],
 ['göründüğü', 'gibi', 'istediğim', 'geldi', 'görseldeki', 'beklediğim', 'fotoğraftaki', 'fotoğrafta', 'gelmedi', 'gözüktüğü'],
 ['elime', 'ulaştı', 'gün', 'içinde', 'kargo', 'günde', 'hızlıydı', 'paketleme', 'ürün', 'sağlam'],
 ['numaranızı', 'numaranizi', 'kalıp', 'alabilirsiniz', 'rahat', 'kendi', 'bedeninizi', 'alın', 'ayakkabı', 'ayak'],
 ['numara', 'giyiyorum', 'ayakkabı', '37', 'rahat', '39', '38', 'normalde', 'tam', 'kalıbı'],
 ['ayakkabı', 'rahat', 'numara', 've', 'numaranızı', 'şık', 'ayak', 'bir', 'numaramı', 'tam'],
 ['gün', 'sonra', 'hiç', 'almayın', 'yok', 'olmadı', 'rağmen', 'oldum', 'zinciri', 'alalı'],
 ['beğendim', 'çok', 'begendim', 'cok', 'begendım', 'beğendimm', 'beğendik', 'dedim', 'yapılan', 'eşim'],
 ['kaliteli', 'kullanışlı', 'fiyatına', 'göre', 'tatlı', 'iyi', 'canta', 'hoş', 'minnoş', 'fiyati'],
 ['elime', 've', 'ulaştı', 'kargo', 'gün', 'çok', 'hızlı', 'paketleme', 'ediyorum', 'ayrıca'],
 ['alin', 'alınabilir', 'bedeninizi', 'alın', 'alınmalı', 'tam', 'kalıp', 'yazlık', 'alinabilir', 'bedeniniz'],
 ['ayakkabı', 'rahat', 'numara', 'tam', 'giyiyorum', '37', 'bir', 'kalıp', '38', 'numaramı'],
 ['fena', 'eder', 'super', 'idare', 'süper', 'dare', 'beğenmedim', 'omuz', 'güzelll', 'anca'],
 ['ama', 'kumaşı', 'iade', 'biraz', 'bana', 'bol', 'ince', 'pantolon', 'iç', 'dar'],
 ['kızım', 'kardeşime', 'begendi', 'beğendi', 'annem', 'aldık', 'hediye', 'eşime', 'kızıma', 'anneme'],
 ['giymeyi', 'dümdüz', 'boyuna', 'koymuşsunuz', 'duracak', 'anlıyorsunuz', 'oradan', 'soru', 'olumsuz', 'kalıplarda'],
 ['güzeller', 'güzell', 'çokk', 'güzelll', 'alacaksanız', 'beyendim', 'düşündüm', 'net', 'güzellll', 'edecek'],
 ['kendi', 'numaranızı', 'alabilirsiniz', 'bedeninizi', 'rahatlığıyla', 'gönül', 'numaranizi', 'ayak', 'kalıp', 'rahat'],
 ['güzeller', 'güzell', 'denedi', 'alacaksanız', 'bolluk', 'salı', 'güzelll', 'alakalı', 'dört', 'sevmiyorum'],
 ['beğendim', 'begendim', 'beğendimmm', 'begendım', 'beğendimm', 'begendimm', 'dışarıdan', 'cok', 'rahatsızlık', 'arkadaki'],
 ['markalar', 'topuğum', 'serçe', 'parmağım', 'zarif', 'oradan', 'diğerini', 'şansım', 'anlıyorsunuz', 'dedikleri'],
 ['büyük', 'bol', 'kumaşı', 'pantolon', 'ama', 'biraz', 'beden', 'dar', 'küçük', 'boyu'],
 ['büyük', 'numara', 'alınmalı', 'kalıbı', 'duruşu', 'küçük', 'alınabilir', 'bi', 'bir', 'alın'],
 ['severek', 'her', 'kullanıyorum', 'gözlük', 'çanta', 'sığıyor', 'bayıldım', 'herkes', 'aldırın', 'şey'],
 ['beğendimm', 'beğendim', 'begendim', 'benimki', 'edecek', '16', 'yazmışlar', 'kalmış', 'alırdım', 'beğendimmm'],
 ['çox', 'bəyəndim', 'harika', 'parmağım', 'fiyatlar', 'mukemmel', 'dedikleri', 'almaya', 'və', 'yapılı'],
 ['38', 'ama', 'ben', 'büyük', 'oldu', 'normalde', 'bir', 'giyiyorum', 'aldım', '37'],
 ['fena', 'gelmedi', 'değil', 'degil', 'beğenmedim', 'fotoğraftaki', 'eder', 'iade', 'kötü', 'gelen'],
 ['harika', 'tek', 'mükemmel', 'kelimeyle', 'her', 'kelime', 'kalite', 'almayın', 'severek', 'kurtarıcı'],
 ['kızım', 'beğenildi', 'kızıma', 'eşime', 'begendi', 'kardeşime', 'aldim', 'kullanıyor', 'beyendi', 'aldık'],
 ['olarak', 'almıştım', 'arkadaşıma', 'kendime', 'anneme', 'beğendi', 'hediye', 'hediyesi', 'aldım', 'günü'],
 ['iyi', 'küpe', 'koku', 'tatlı', 'zarif', 'ucu', 'hoş', 'fiyata', 'kaliteli', 'kolye'],
 ['gayet', 'ürün', 'ürünler', 'fp', 'urun', 'güzel', 'ürünüm', 'sağlam', 'fiyatlı', 'güvenli'],
 ['gayet', 'fiyatına', 'ürün', 'ürünler', 'göre', 'sağlam', 'sorunsuz', 'teşekkürler', 'teslimat', 'urun'],
 ['rahat', 'ayakkabı', 'numara', 'giyiyorum', '37', 'duruyor', '38', '39', 'çok', 'numaramı'],
 ['teşekkür', 'ederim', 'hediyeniz', 'ayrıca', 'paketleme', 'özenli', 'içinde', 'satıcıya', 'teşekkürler', 'hediyeler'],
 ['kaliteli', 'tavsiye', 'herkese', 'ediyorum', 'kullanışlı', 'özenli', 'paketleme', 'gözlük', 'sağlık', 'hediyeniz'],
 ['ürünü', 'fiyat', 'performans', 'rahatlığıyla', 'gönül', 'istediğim', 'alabilirsiniz', 'performansı', 'numaranızı', 'aynısı'],
 ['fiyata', 'gerçekten', 'kaliteli', 'ediyorum', 'iyi', 'çanta', 'kalitesi', 've', 'paketleme', 'gözlük'],
 ['günlük', 'gün', 'kullanım', 'günde', 'güneş', 'kullanıma', 'günlerde', 'gun', 'güncellerim', 'günden'],
 ['kardeşime', 'kızım', 'kızıma', 'begendi', 'beğenildi', 'beğendi', 'aldim', 'aldık', 'kız', 'anneme'],
 ['hediyesi', 'günlük', 'günü', 'doğum', 'kullanım', 'uygun', 'umarım', 'ideal', 'kararma', 'kullanıma'],
 ['şık', 'rahat', 'tavsiye', 'duruyor', 've', 'ayakkabı', 'tatlı', 'çok', 'zarif', 'ayakta']

]

# Bütün kelimeleri tek bir liste haline getir
all_words = []
for topic in topics_words:
    all_words.extend(topic)

# Kelimeleri tek bir string haline getir
text = ' '.join(all_words)

# Word Cloud oluştur
wordcloud = WordCloud(width=1600, height=800, background_color='white', mask=mask, contour_width=3, contour_color='black').generate(text)

# Görselleştir
plt.figure(figsize=(16, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of CTM Topics (135K Reviews)', fontsize=20)
plt.show()
