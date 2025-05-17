# Google Cloud Platform Entegrasyonu - Detaylı Talimatlar

Bu belgede, AI Face Detection projesinin Google Cloud Platform (GCP) ile nasıl entegre edileceği detaylı olarak açıklanmaktadır.

## 1. Kurulum Adımları

### 1.1. Google Cloud SDK Kurulumu

Eğer Google Cloud SDK'yi henüz kurmadıysanız:

1. [Google Cloud SDK indirme sayfasından](https://cloud.google.com/sdk/docs/install) işletim sisteminize uygun kurulum dosyasını indirin.
2. Kurulum sırasında "Add to PATH" seçeneğini işaretleyin.
3. Kurulum tamamlandıktan sonra, bir PowerShell veya Komut İstemi penceresi açın ve şu komutla kontrol edin:
   ```
   gcloud --version
   ```

### 1.2. Kimlik Doğrulama ve Proje Ayarları

1. Komut satırında aşağıdaki komutu çalıştırarak Google hesabınızla kimlik doğrulaması yapın:

   ```
   gcloud auth application-default login
   ```

   Bu komut bir tarayıcı penceresi açacak ve Google hesabınızla giriş yapmanızı isteyecek.

2. Projenizi etkin olarak ayarlayın:
   ```
   gcloud config set project wingie-devops-project
   ```

### 1.3. GCP Bucket Yapılandırması

GCP bucket ve klasörlerini kontrol etmek için:

1. Bucket'ınızı kontrol edin:

   ```
   gsutil ls
   ```

2. Bucket içindeki klasörleri kontrol edin:

   ```
   gsutil ls gs://distinguish-ai-faces-dataset/
   ```

3. Klasörler içindeki dosyaları kontrol edin:

   ```
   gsutil ls gs://distinguish-ai-faces-dataset/ai-faces/
   gsutil ls gs://distinguish-ai-faces-dataset/human-faces/
   ```

4. Klasör yapısı oluşturmanız gerekirse:
   ```
   gsutil cp NUL gs://distinguish-ai-faces-dataset/ai-faces/
   gsutil cp NUL gs://distinguish-ai-faces-dataset/human-faces/
   ```

### 1.4. Kimlik Bilgilerini Ayarlama

Kimlik bilgilerinin konumu genellikle şurada bulunur:

- Windows: `%APPDATA%\gcloud\application_default_credentials.json`
- Linux/Mac: `~/.config/gcloud/application_default_credentials.json`

Bu dosyanın konumunu ortam değişkenine ekleyin:

```powershell
# Windows PowerShell
$env:GOOGLE_APPLICATION_CREDENTIALS="$env:APPDATA\gcloud\application_default_credentials.json"
```

## 2. Python Bağımlılıklarını Yükleme

Google Cloud Storage Python paketini yüklemek için:

```
pip install google-cloud-storage
```

## 3. Veri Yapısını Hazırlama

GCP bucket'ınızda verilerinizi şu yapıda düzenlemeniz gerekiyor:

```
distinguish-ai-faces-dataset/
  ├── ai-faces/       # Yapay zeka üretimi yüzler resimleri (.jpg)
  └── human-faces/    # Gerçek insan yüzleri resimleri (.jpg)
```

## 4. Otomatik Kurulum Betiği

Tüm kurulum adımlarını otomatikleştiren bir PowerShell betiği hazırladık:

```
./training_pipeline/setup_gcp.ps1
```

Bu betik aşağıdakileri yapacaktır:

- Google Cloud SDK'nin konumunu bulma
- Kimlik doğrulaması yapma
- Proje ayarlarını yapılandırma
- Bucket'ları listeleme
- Gerekli Python paketlerini kontrol etme ve kurma

## 5. Proje Yapılandırması

GCP entegrasyonu varsayılan olarak `src/config.py` dosyasında etkinleştirilmiştir:

```python
GCP_CONFIG = {
    'use_gcp': True,  # GCP kullanımı aktif
    'bucket_name': os.environ.get('GCP_BUCKET_NAME', 'distinguish-ai-faces-dataset'),
    'ai_folder': 'ai-faces',
    'human_folder': 'human-faces'
}
```

## 6. GCP Entegrasyonunu Test Etme

Kurulumu test etmek için şu komutu çalıştırın:

```
python main.py --model efficientnet
```

Bunun şu işlemleri yapması gerekiyor:

1. GCP bucket'tan veri indirme
2. Verileri yerel olarak önişleme
3. Modeli eğitme
4. Sonuçları değerlendirme

## 7. Sorun Giderme

### 7.1. "No Module Named 'google.cloud'" Hatası

Bu hata, Python bağımlılıklarının eksik olduğunu gösterir:

```
pip install google-cloud-storage
```

### 7.2. Yetkilendirme Hataları

"Permission denied" hataları için:

1. Doğru Google hesabıyla giriş yaptığınızdan emin olun.
2. Hesabınızın projeye ve Storage'a erişim izni olduğunu kontrol edin.

```
gcloud projects get-iam-policy wingie-devops-project
```

### 7.3. Kimlik Bilgisi Hataları

"Could not automatically determine credentials" hatası için:

1. Kimlik doğrulamayı yeniden yapın:
   ```
   gcloud auth application-default login
   ```
2. GOOGLE_APPLICATION_CREDENTIALS ortam değişkeninin doğru ayarlandığından emin olun.

### 7.4. Bucket veya Veri Erişimi Sorunları

1. Bucket'ın varlığını kontrol edin:
   ```
   gsutil ls
   ```
2. Bucket içindeki klasörleri kontrol edin:
   ```
   gsutil ls gs://distinguish-ai-faces-dataset/
   ```
3. Klasörlerdeki dosyaları kontrol edin:
   ```
   gsutil ls gs://distinguish-ai-faces-dataset/ai-faces/
   gsutil ls gs://distinguish-ai-faces-dataset/human-faces/
   ```

## 8. Verilerinizi Yükleme

GCP bucket'a veri yüklemek için:

```
gsutil cp "yerel/dosya/yolu/resim.jpg" gs://distinguish-ai-faces-dataset/ai-faces/
gsutil cp "yerel/dosya/yolu/resim.jpg" gs://distinguish-ai-faces-dataset/human-faces/
```

Bir klasördeki tüm dosyaları yüklemek için:

```
gsutil -m cp "yerel/klasör/yolu/*.jpg" gs://distinguish-ai-faces-dataset/ai-faces/
gsutil -m cp "yerel/klasör/yolu/*.jpg" gs://distinguish-ai-faces-dataset/human-faces/
```
