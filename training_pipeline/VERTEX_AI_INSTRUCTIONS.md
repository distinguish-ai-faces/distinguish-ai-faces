# Vertex AI ile AI Yüz Tespiti Modeli Eğitimi

Bu belge, AI yüz tespiti modelinizi Google Cloud Vertex AI platformunda nasıl eğiteceğinizi açıklar.

## İçindekiler

1. [Gereksinimler](#gereksinimler)
2. [Kurulum](#kurulum)
3. [Konfigürasyon](#konfigürasyon)
4. [Docker Container'ı Oluşturma](#docker-containerı-oluşturma)
5. [Vertex AI Eğitim İşi Başlatma](#vertex-ai-eğitim-işi-başlatma)
6. [Sonuçları İzleme ve Modeli İndirme](#sonuçları-izleme-ve-modeli-indirme)
7. [Sorun Giderme](#sorun-giderme)

## Gereksinimler

Bu entegrasyonu kullanmak için aşağıdaki bileşenlere ihtiyacınız vardır:

- **Google Cloud Hesabı** ve aktif bir **Google Cloud Projesi**
- **Google Cloud SDK** (gcloud CLI)
- **Docker** (Docker Desktop veya Docker Engine)
- **Python 3.8+** ve aşağıdaki paketler:
  - google-cloud-aiplatform
  - google-cloud-storage
  - pyyaml

## Kurulum

### 1. İlk Kurulum (Windows)

Windows'ta kolay kurulum için, sağlanan PowerShell betiğini çalıştırın:

```powershell
./setup_vertex_ai.ps1
```

Bu betik şunları yapacaktır:

- Gerekli Python paketlerini kontrol edip kurma
- Google Cloud SDK'yi kontrol etme
- Google Cloud kimlik doğrulaması yapma
- Proje ayarlarını yapılandırma
- Gerekli API'leri etkinleştirme
- Docker'ı Google Cloud ile yapılandırma
- GCP bucket'ları kontrol etme ve gerekiyorsa oluşturma
  - `distinguish-ai-faces-dataset` (veri için)
  - `distinguish-ai-faces-model` (model için)

### 2. Manuel Kurulum (Linux/macOS)

Linux veya macOS kullanıyorsanız, aşağıdaki adımları takip edin:

```bash
# Google Cloud SDK'yi kurun (eğer yüklü değilse)
# https://cloud.google.com/sdk/docs/install adresinden kurulum talimatlarını izleyin

# Kimlik doğrulaması yapın
gcloud auth application-default login

# Projenizi ayarlayın
gcloud config set project wingie-devops-project

# Gerekli API'leri etkinleştirin
gcloud services enable aiplatform.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Docker'ı gcloud ile yapılandırın
gcloud auth configure-docker

# Gerekli Python paketlerini yükleyin
pip install google-cloud-aiplatform google-cloud-storage pyyaml

# Bucket'ları kontrol edin ve oluşturun (yoksa)
gsutil ls gs://distinguish-ai-faces-dataset || gsutil mb gs://distinguish-ai-faces-dataset
gsutil ls gs://distinguish-ai-faces-model || gsutil mb gs://distinguish-ai-faces-model
```

## Konfigürasyon

Vertex AI eğitiminizi yapılandırmak için, `vertex_ai_config.yaml` dosyasını düzenleyin:

```yaml
# GCP Project Configuration
project_id: "wingie-devops-project" # GCP Project ID
region: "us-central1" # GCP Region

# Training Container Configuration
container:
  image_uri: "gcr.io/wingie-devops-project/ai-face-detection:latest"
  env_vars:
    - name: "DATA_BUCKET_NAME"
      value: "distinguish-ai-faces-dataset"
    - name: "MODEL_BUCKET_NAME"
      value: "distinguish-ai-faces-model"

# Machine Configuration
machine_config:
  machine_type: "n1-standard-8" # 8 vCPUs, 30 GB memory
  accelerator_type: "NVIDIA_TESLA_T4" # GPU type
  accelerator_count: 1 # Number of GPUs

# Storage Configuration
staging_bucket: "gs://distinguish-ai-faces-dataset" # Veri staging için
output_uri_prefix: "gs://distinguish-ai-faces-model/vertex_output" # Model çıktıları için

# Training Configuration
training_params:
  model_type: "efficientnet" # Model mimarisi
  epochs: 20 # Eğitim dönemleri
  batch_size: 32 # Batch boyutu
  img_size: # Görüntü boyutu
    - 224
    - 224
  learning_rate: 0.0001 # Öğrenme oranı
```

Farklı makine türleri ve GPU'lar için seçenekler:

| Makine Türü    | Açıklama           |
| -------------- | ------------------ |
| n1-standard-4  | 4 vCPU, 15 GB RAM  |
| n1-standard-8  | 8 vCPU, 30 GB RAM  |
| n1-standard-16 | 16 vCPU, 60 GB RAM |
| n1-highmem-8   | 8 vCPU, 52 GB RAM  |

| Hızlandırıcı Türü | Açıklama                          |
| ----------------- | --------------------------------- |
| NVIDIA_TESLA_T4   | Maliyet etkin GPU                 |
| NVIDIA_TESLA_V100 | Yüksek performanslı GPU           |
| NVIDIA_TESLA_P100 | Performans/maliyet dengesinde GPU |
| NVIDIA_TESLA_P4   | Giriş seviyesi GPU                |

## Docker Container'ı Oluşturma

Eğitim kodunuzu içeren bir Docker container'ı oluşturmak ve Google Container Registry'ye push etmek için:

### Windows:

```bash
# Bash uyumlu terminal kullanın (Git Bash, WSL, vb.)
bash build_push_container.sh
```

### Linux/macOS:

```bash
# Betik dosyasını çalıştırılabilir yapın
chmod +x build_push_container.sh

# Betiği çalıştırın
./build_push_container.sh
```

## Vertex AI Eğitim İşi Başlatma

### Temel Kullanım:

```bash
python run_vertex_training.py
```

Bu komut, `vertex_ai_config.yaml` dosyasındaki yapılandırmayı kullanarak bir eğitim işi başlatacaktır.

### Özel Parametrelerle Kullanım:

```bash
python run_vertex_training.py --model resnet50 --epochs 30 --batch-size 16
```

### Kullanılabilir Parametreler:

- `--config`: Yapılandırma YAML dosyasının yolu
- `--model`: Model mimarisini geçersiz kılma (efficientnet, resnet50, mobilenet, scratch)
- `--epochs`: Eğitim dönemlerinin sayısını geçersiz kılma
- `--batch-size`: Batch boyutunu geçersiz kılma
- `--job-name`: Özel bir iş adı belirleme

## Sonuçları İzleme ve Modeli İndirme

### Google Cloud Console'dan İzleme:

1. [Vertex AI Dashboard](https://console.cloud.google.com/vertex-ai) sayfasına gidin
2. Sol menüden "Training" seçeneğine tıklayın
3. Eğitim işinizi listeden bulun ve detayları görmek için tıklayın
4. "Logs" sekmesinden eğitim günlüklerini izleyebilirsiniz

### Eğitilmiş Modeli İndirme:

Eğitim tamamlandıktan sonra, model `gs://distinguish-ai-faces-model/` yolundaki GCP bucket'ınıza kaydedilecektir.

```bash
# Vertex çıktı klasörünü listeleyin
gsutil ls gs://distinguish-ai-faces-model/vertex_output/

# Belirli bir model çıktısını listeleyin
gsutil ls gs://distinguish-ai-faces-model/vertex_output/[MODEL_ADI]/

# Modeli Google Cloud Storage'dan indirme
gsutil cp gs://distinguish-ai-faces-model/vertex_output/[MODEL_ADI]/best_model.h5 ./downloaded_model.h5
```

### Bucket Yapısı:

- **distinguish-ai-faces-dataset**: Veri dosyalarını içerir

  - `/ai-faces`: AI yüzleri içeren görüntüler
  - `/human-faces`: İnsan yüzleri içeren görüntüler

- **distinguish-ai-faces-model**: Model dosyalarını içerir
  - `/vertex_output`: Vertex AI eğitim çıktıları
  - `/vertex_output/[MODEL_ADI]`: Eğitilmiş modeller ve değerlendirme sonuçları

## Sorun Giderme

### Yaygın Hatalar ve Çözümleri:

#### 1. Docker Hataları

**Hata**: "Docker command not found" veya "Docker daemon not running"

**Çözüm**:

- Docker'ın yüklü ve çalışır durumda olduğundan emin olun
- Windows'ta Docker Desktop'ın çalıştığını kontrol edin
- Linux'ta Docker servisini başlatın: `sudo systemctl start docker`

#### 2. Google Cloud Yetkilendirme Hataları

**Hata**: "Could not determine the project ID" veya "Permission denied"

**Çözüm**:

- Kimlik doğrulaması yapın: `gcloud auth application-default login`
- Doğru projeyi seçin: `gcloud config set project YOUR_PROJECT_ID`
- IAM izinlerinizi kontrol edin (en az "Vertex AI User" ve "Storage Admin" rolleri gereklidir)

#### 3. GPU Erişim Sorunları

**Hata**: "No GPU quota" veya "The zone does not have enough resources"

**Çözüm**:

- [Quota sayfasından](https://console.cloud.google.com/iam-admin/quotas) GPU kotanızı kontrol edin
- Farklı bir bölge veya GPU türü seçin
- Daha küçük bir makine türüyle veya GPU olmadan deneme yapın

### Yardım Alma:

Eğer yukarıdaki çözümler işe yaramazsa:

- Google Cloud'un [resmi dokümantasyonuna](https://cloud.google.com/vertex-ai/docs/training/custom-training) bakın
- [Google Cloud Support](https://cloud.google.com/support) ile iletişime geçin
- [Stack Overflow](https://stackoverflow.com/questions/tagged/google-cloud-vertex-ai) üzerinde "google-cloud-vertex-ai" etiketiyle soru sorun

## Ek Kaynaklar

- [Vertex AI Dokümantasyonu](https://cloud.google.com/vertex-ai/docs)
- [Custom Training Guide](https://cloud.google.com/vertex-ai/docs/training/custom-training)
- [Vertex AI Python API Referansı](https://googleapis.dev/python/aiplatform/latest/index.html)
- [Google Cloud Container Registry Dokümantasyonu](https://cloud.google.com/container-registry/docs)
