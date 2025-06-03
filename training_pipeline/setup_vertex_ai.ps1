# Vertex AI için hazırlık PowerShell betiği

# Gerekli paketleri kontrol et ve yükle
Write-Host "Google Cloud SDK ve Python paketleri kontrol ediliyor..."

# Google Cloud paketini kontrol et
try {
    python -c "from google.cloud import aiplatform; print('Google Cloud AI Platform paketi yüklü.')"
} catch {
    Write-Host "Google Cloud AI Platform paketi yükleniyor..."
    pip install google-cloud-aiplatform
}

try {
    python -c "import yaml; print('PyYAML paketi yüklü.')"
} catch {
    Write-Host "PyYAML paketi yükleniyor..."
    pip install pyyaml
}

# Google Cloud SDK yolunu kontrol et
$gcloudPath = "gcloud"
try {
    $gcloudVersion = & $gcloudPath version
    Write-Host "Google Cloud SDK bulundu!"
    Write-Host $gcloudVersion[0]
} catch {
    Write-Host "Google Cloud SDK PATH'te bulunamadı. Alternatif yolları kontrol ediyoruz..."
    
    # Alternatif yolları kontrol et
    $possiblePaths = @(
        "C:\Program Files (x86)\Google\Cloud SDK\bin\gcloud.cmd",
        "C:\Program Files\Google\Cloud SDK\bin\gcloud.cmd",
        "C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $gcloudPath = $path
            Write-Host "Google Cloud SDK bulundu: $gcloudPath"
            break
        }
    }
}

# gsutil yolunu ayarla
$gsutilPath = $gcloudPath -replace "gcloud", "gsutil"
if (-not ($gsutilPath -match "gsutil")) {
    $gsutilPath = $gsutilPath.Substring(0, $gsutilPath.LastIndexOf('\')) + "\gsutil.cmd"
}

# Kimlik doğrulaması ve proje ayarları
Write-Host "`nGoogle Cloud kimlik doğrulaması yapılıyor..."
try {
    & $gcloudPath auth application-default login
} catch {
    Write-Host "Kimlik doğrulama hatası: $_"
    Write-Host "Lütfen manuel olarak şu komutu çalıştırın: gcloud auth application-default login"
    Exit 1
}

Write-Host "`nAktif proje 'wingie-devops-project' olarak ayarlanıyor..."
try {
    & $gcloudPath config set project wingie-devops-project
} catch {
    Write-Host "Proje ayarlama hatası: $_"
    Exit 1
}

# Vertex AI API'sini etkinleştir
Write-Host "`nVertex AI API'sini etkinleştiriyoruz..."
try {
    & $gcloudPath services enable aiplatform.googleapis.com
} catch {
    Write-Host "API etkinleştirme hatası: $_"
    Exit 1
}

# Container Registry API'sini etkinleştir
Write-Host "`nContainer Registry API'sini etkinleştiriyoruz..."
try {
    & $gcloudPath services enable containerregistry.googleapis.com
} catch {
    Write-Host "API etkinleştirme hatası: $_"
    Exit 1
}

# Docker'ın yüklü olup olmadığını kontrol et
Write-Host "`nDocker kurulumu kontrol ediliyor..."
try {
    docker --version
    Write-Host "Docker kurulu!"
} catch {
    Write-Host "Docker kurulu değil veya PATH'te bulunamadı."
    Write-Host "Lütfen Docker Desktop'ı yükleyin: https://www.docker.com/products/docker-desktop"
    Exit 1
}

# Docker'ı gcloud ile yapılandır
Write-Host "`nDocker'ı Google Cloud ile yapılandırıyoruz..."
try {
    & $gcloudPath auth configure-docker
} catch {
    Write-Host "Docker yapılandırma hatası: $_"
    Exit 1
}

# GCP bucket'ları kontrol edip yoksa oluştur
Write-Host "`nGCP bucket'ları kontrol ediliyor..."

# Veri bucket'ı kontrol et
$dataBucketName = "distinguish-ai-faces-dataset"
try {
    $bucketExists = & $gsutilPath ls "gs://$dataBucketName" 2>$null
    if ($null -eq $bucketExists) {
        Write-Host "Veri bucket'ı bulunamadı. Oluşturuluyor: $dataBucketName"
        & $gsutilPath mb "gs://$dataBucketName"
        
        # Veri klasörlerini oluştur
        Write-Host "Veri klasörleri oluşturuluyor..."
        & $gsutilPath cp NUL "gs://$dataBucketName/ai-faces/"
        & $gsutilPath cp NUL "gs://$dataBucketName/human-faces/"
    } else {
        Write-Host "Veri bucket'ı mevcut: $dataBucketName"
    }
} catch {
    Write-Host "Veri bucket'ı kontrolünde hata: $_"
}

# Model bucket'ı kontrol et
$modelBucketName = "distinguish-ai-faces-model"
try {
    $bucketExists = & $gsutilPath ls "gs://$modelBucketName" 2>$null
    if ($null -eq $bucketExists) {
        Write-Host "Model bucket'ı bulunamadı. Oluşturuluyor: $modelBucketName"
        & $gsutilPath mb "gs://$modelBucketName"
        
        # Model klasörlerini oluştur
        Write-Host "Model klasörleri oluşturuluyor..."
        & $gsutilPath cp NUL "gs://$modelBucketName/vertex_output/"
    } else {
        Write-Host "Model bucket'ı mevcut: $modelBucketName"
    }
} catch {
    Write-Host "Model bucket'ı kontrolünde hata: $_"
}

Write-Host "`nVertex AI kurulumu tamamlandı. Şimdi yapabileceğiniz işlemler:"
Write-Host "1. Docker container'ı build edip push etmek için:"
Write-Host "   ./build_push_container.sh"
Write-Host "2. Vertex AI eğitimini başlatmak için:"
Write-Host "   python run_vertex_training.py"
Write-Host "3. Farklı model mimarisi kullanmak için:"
Write-Host "   python run_vertex_training.py --model resnet50 --epochs 30" 