# AI Face Detection Training Pipeline

Bu modül, yapay zeka ile üretilmiş yüzleri gerçek insan yüzlerinden ayırt etmek için bir derin öğrenme modeli eğitmek için kullanılır.

## Özellikler

- Yapay zeka ile üretilmiş yüzler ve gerçek insan yüzleri üzerinde model eğitimi
- Farklı model mimarileri desteği (EfficientNet, ResNet, MobileNet, özel CNN)
- Veri yükleme, ön işleme ve artırma
- Google Cloud Storage entegrasyonu
- Model değerlendirme araçları
- Görselleştirme ve metrik raporlama

## Kurulum

### Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

## Kullanım

### Komut Satırı Arayüzü

```bash
python main.py --model efficientnet --img-size 224 224 --batch-size 32 --epochs 20
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

### GCP Konfigürasyonu

Google Cloud Storage kullanmak için:

1. GCP kimlik bilgilerini ayarlayın

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
   ```

2. Bucket adını belirtin

   ```bash
   export GCP_BUCKET_NAME=your-bucket-name
   ```

3. GCP kullanımını etkinleştirin
   ```bash
   python main.py --use-gcp --gcp-bucket your-bucket-name
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
outputs/
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

## Model Mimarileri

### EfficientNetB0 (Varsayılan)

EfficientNetB0, yüksek doğruluk ve verimlilik için optimize edilmiş bir ön eğitimli modeldir. Daha küçük model boyutu ile iyi performans sağlar.

### ResNet50

Daha derin bir model mimarisi olan ResNet50, karmaşık kalıpları öğrenmek için daha fazla kapasite sunar.

### MobileNetV2

MobileNetV2, mobil ve kenar cihazlar için optimize edilmiş hafif bir modeldir. Daha hızlı çıkarım süreleri sağlar.

### Özel CNN (scratch)

Temel bir evrişimli sinir ağı modelidir. İlgili veri kümesine özel olarak eğitilir.

## Performans

Modellerin karşılaştırmalı performansı:

| Model          | Doğruluk | ROC AUC | PR AUC | Eğitim Süresi |
| -------------- | -------- | ------- | ------ | ------------- |
| EfficientNetB0 | ~95%     | ~0.98   | ~0.97  | Orta          |
| ResNet50       | ~94%     | ~0.97   | ~0.96  | Yüksek        |
| MobileNetV2    | ~92%     | ~0.95   | ~0.94  | Düşük         |
| Özel CNN       | ~88%     | ~0.92   | ~0.90  | Orta          |

_Not: Gerçek performans, veri kümesi ve eğitim parametrelerine bağlı olarak değişecektir._
