Teknik Özet ve Bulgular
Raporumda yer alan kapsamlı analizlerin özeti aşağıdadır:

Algoritma Karşılaştırması: Adam algoritması, momentum özelliği sayesinde salınımları engelleyerek en kararlı ve hızlı yakınsayan yöntem olmuştur.

Zaman Verimliliği: Adam algoritması, hatayı (loss) 0.1 seviyesine yaklaşık 29 saniyede indirirken; GD algoritması benzer sürede ancak 0.4 seviyesine ulaşabilmiştir.

Overfitting Analizi: Eğitim hatası başarıyla düşmesine rağmen test başarısının %50 civarında sabitlenmesi, modelin veriyi genelleştirmek yerine ezberlediğini (overfitting) göstermektedir.

Görselleştirme: T-SNE analizi ile ağırlık parametrelerinin optimizasyon yörüngeleri 2D uzayda modellenmiştir.
 
 Teknik Detaylar

Aktivasyon Fonksiyonu: Modelde tanh fonksiyonu kullanılmıştır.

Vektör İşlemleri: Tüm vektörel hesaplamalar (çarpım, toplama, çıkarma) kütüphane bağımsız olarak manuel kodlanmıştır.
Tam Rapor
Matematiksel türetimlerin ve detaylı grafik yorumlarının yer aldığı teknik raporun tamamına buradan ulaşabilirsiniz:
[Raporu Görüntüle](./rapor.pdf)
