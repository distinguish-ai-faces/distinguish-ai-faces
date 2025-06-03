# AI Yüz Tespiti Eğitim Modülü

Bu modül, yapay zeka ile üretilmiş yüzleri gerçek insan yüzlerinden ayırt etmek için bir derin öğrenme modeli eğitmek için kullanılır.

## Geliştirme Süreci ve Sonuçlar

### Model Sürümleri ve Performans

Projede şu ana kadar 5 farklı model sürümü geliştirilmiştir:

| Sürüm | Model Mimarisi | Doğruluk | ROC AUC | PR AUC | Notlar                                 |
| ----- | -------------- | -------- | ------- | ------ | -------------------------------------- |
| v2    | EfficientNet   | %86.89   | 0.9083  | 0.9212 | En iyi performans gösteren model       |
| v3    | EfficientNet   | %71.31   | 0.8429  | 0.8313 | Veri artırma stratejileri geliştirildi |
| v4    | EfficientNet   | %77.87   | 0.9225  | 0.9350 | Gelişmiş ön işleme eklendi             |
| v5    | EfficientNet   | %54.92   | 0.5414  | 0.5848 | Aşırı öğrenme sorunu yaşandı           |

### Kullanılan Veri Artırma Teknikleri

- Rastgele döndürme (40 dereceye kadar)
- Genişlik/yükseklik kaydırma (%30'a kadar)
- Kesme (shear) dönüşümleri (%30'a kadar)
- Yakınlaştırma/uzaklaştırma (%30'a kadar)
- Yatay çevirme
- Parlaklık ayarlamaları (%70-%130 arasında)
- Renk kanalı kaydırmaları
- Rastgele Gaussian gürültüsü ekleme
- Kontrast ayarlamaları

### Gelişmiş Ön İşleme Adımları

- Görüntü normalizasyonu (0-1 aralığına)
- Boyut standardizasyonu (224x224)
- Batch işleme optimizasyonu
- Veri ön yükleme (prefetch) ile performans iyileştirme

### Eğitim Stratejileri

- Erken durdurma (Early stopping) ile aşırı öğrenmenin önlenmesi
- Öğrenme oranı azaltma (ReduceLROnPlateau)
- Checkpoint kaydı ile en iyi modellerin saklanması
- Eğitim/doğrulama/test veri seti ayrımı (%70/%20/%10)
- Batch normalizasyonu
- Dropout (%60) ile düzenlileştirme

## Özellikler

- Yapay zeka ile üretilmiş yüzler ve gerçek insan yüzleri üzerinde model eğitimi
- Farklı model mimarileri desteği (EfficientNet, ResNet, MobileNet, özel CNN)
- Veri yükleme, ön işleme ve artırma
- Google Cloud Storage entegrasyonu
- Vertex AI ile bulut tabanlı eğitim
- Model değerlendirme araçları
- Görselleştirme ve metrik raporlama
- Ensemble tahmin ile çoklu model kullanımı
- Webcam ile gerçek zamanlı test

## Kurulum

### Bağımlılıkları Yükleme

```bash
# Windows için PowerShell script ile kurulum
.\install_dependencies.ps1

# veya manuel olarak
pip install -r requirements.txt
```

### GCP Kurulumu (İsteğe Bağlı)

```bash
# GCP kurulum script'ini çalıştır
.\setup_gcp.ps1

# Vertex AI kurulum script'ini çalıştır
.\setup_vertex_ai.ps1
```

## Kullanım

### Yerel Eğitim

```bash
python main.py --model efficientnet --img-size 224 224 --batch-size 32 --epochs 20 --output-dir outputs_v6
```

### Vertex AI ile Eğitim

```bash
python run_vertex_training.py --config vertex_ai_config.yaml
```

### Tahmin Yapma

```bash
# Tek model ile tahmin
python predict_cli.py --model outputs_v4/final_model.h5 --image test_image.jpg

# Ensemble tahmin (birden fazla model)
python ensemble_predict_cli.py --model-versions v2 v3 v4 --image test_image.jpg

# Webcam ile gerçek zamanlı test
python webcam_test.py --model outputs_v4/final_model.h5
```

### Kullanılabilir Seçenekler

- `--model`: Kullanılacak model mimarisi (`efficientnet`, `resnet50`, `mobilenet`, `scratch`)
- `--data-dir`: Veri dizini
- `--img-size`: Giriş görüntü boyutu
- `--batch-size`: Eğitim için yığın boyutu
- `--use-gcp`: Google Cloud Storage kullanımını etkinleştirir
- `--gcp-bucket`: GCP bucket adı
- `--epochs`: Eğitim dönemleri sayısı
- `--output-dir`: Model ve günlükler için çıktı dizini
- `--learning-rate`: Öğrenme oranı (varsayılan: 0.0001)

## Model Mimarileri

### EfficientNetB0 (Varsayılan)

EfficientNetB0, yüksek doğruluk ve verimlilik için optimize edilmiş bir ön eğitimli modeldir. Daha küçük model boyutu ile iyi performans sağlar. Projede en iyi sonuçları bu model ile elde ettik.

Konfigürasyon:

- Dropout oranı: 0.6
- Taban model dondurma: Hayır (fine-tuning yapılıyor)
- Sabır (patience): 8 epoch
- Minimum delta: 0.005

### ResNet50

Daha derin bir model mimarisi olan ResNet50, karmaşık kalıpları öğrenmek için daha fazla kapasite sunar.

Konfigürasyon:

- Dropout oranı: 0.6
- Taban model dondurma: Hayır (fine-tuning yapılıyor)
- Sabır (patience): 8 epoch
- Minimum delta: 0.005

### MobileNetV2

MobileNetV2, mobil ve kenar cihazlar için optimize edilmiş hafif bir modeldir. Daha hızlı çıkarım süreleri sağlar.

Konfigürasyon:

- Dropout oranı: 0.6
- Taban model dondurma: Hayır (fine-tuning yapılıyor)
- Sabır (patience): 8 epoch
- Minimum delta: 0.005

### Özel CNN (scratch)

Temel bir evrişimli sinir ağı modelidir. İlgili veri kümesine özel olarak eğitilir.

Mimari:

- 4 konvolüsyonel blok (32, 64, 128, 256 filtre)
- Her blokta batch normalizasyon ve dropout
- 512 ve 128 nöronlu tam bağlantılı katmanlar
- Sigmoid aktivasyonlu çıkış katmanı

## Ensemble Tahmin Sistemi

Birden fazla modelin tahminlerini birleştirerek daha güvenilir sonuçlar elde etmek için ensemble tahmin sistemi geliştirilmiştir. Bu sistem:

1. Farklı model sürümlerini yükler
2. Her model için ayrı tahmin yapar
3. Tahminleri ağırlıklı ortalama ile birleştirir
4. Sonuçları görselleştirir

```bash
python ensemble_predict_cli.py --model-versions v2 v4 --image test.jpg --output-dir results
```

## Veri Yapısı

Beklenen veri dizin yapısı:

```
data/
  ├── ai_faces/       # Yapay zeka üretimi yüzler
  │   └── *.jpg
  │
  └── human_faces/    # Gerçek insan yüzleri
      └── *.jpg
```

## Eğitim Çıktıları

Eğitim çıktıları şu yapıda kaydedilir:

```
outputs_vX/
  ├── checkpoints/          # Model kontrol noktaları
  │   ├── model_XX_YYYY.h5  # Dönem ve doğruluk ile kaydedilen modeller
  │   └── best_model.h5     # En iyi model
  │
  ├── logs/                 # TensorBoard günlükleri
  │
  ├── evaluation/           # Değerlendirme çıktıları
  │   ├── confusion_matrix.png
  │   ├── roc_curve.png
  │   ├── precision_recall_curve.png
  │   └── evaluation_results.json
  │
  └── final_model.h5        # Son model
```

## Gelecek Çalışmalar

- Transfer öğrenme stratejilerinin geliştirilmesi
- Daha büyük ve çeşitli veri kümesi oluşturma
- Daha derin model mimarileri deneme
- Hyperparameter optimizasyonu
- Açıklanabilir AI teknikleri ile model yorumlanabilirliğini artırma
