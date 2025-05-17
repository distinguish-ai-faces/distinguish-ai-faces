# Google Cloud kimlik bilgilerini ayarlayan PowerShell betiği

# SDK bin dizinini tanımlayın - eğer bu yol çalışmazsa alternatif yolları kontrol edin
$gcloudPath = "gcloud"
$gsutilPath = "gsutil"

# SDK'nin PATH'te olup olmadığını kontrol edin
Write-Host "Google Cloud SDK yolu kontrol ediliyor..."
try {
    $gcloudVersion = & $gcloudPath version
    Write-Host "Google Cloud SDK bulundu!"
    Write-Host $gcloudVersion[0]
} catch {
    Write-Host "Google Cloud SDK PATH'te bulunamadı. Alternatif yolları kontrol ediyoruz..."
    
    # Alternatif yolları kontrol edin
    $possiblePaths = @(
        "C:\Program Files (x86)\Google\Cloud SDK\bin\gcloud.cmd",
        "C:\Program Files\Google\Cloud SDK\bin\gcloud.cmd",
        "C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd",
        "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    )
    
    foreach ($path in $possiblePaths) {
        if (Test-Path $path) {
            $gcloudPath = $path
            $gsutilPath = $path -replace "gcloud.cmd", "gsutil.cmd"
            Write-Host "Google Cloud SDK bulundu: $gcloudPath"
            break
        }
    }
}

# Kullanıcı kimlik doğrulaması yapın (tarayıcıyı açar)
Write-Host "Google Cloud kimlik doğrulaması yapılıyor..."
try {
    & $gcloudPath auth application-default login
} catch {
    Write-Host "Kimlik doğrulama hatası: $_"
    Write-Host "Lütfen manuel olarak şu komutu çalıştırın: gcloud auth application-default login"
}

# Aktif projeyi ayarlayın
Write-Host "Aktif proje 'wingie-devops-project' olarak ayarlanıyor..."
try {
    & $gcloudPath config set project wingie-devops-project
} catch {
    Write-Host "Proje ayarlama hatası: $_"
}

# GCP bucket varsa listeleyin
Write-Host "Mevcut bucket'lar listeleniyor..."
try {
    & $gsutilPath ls
} catch {
    Write-Host "Bucket listeleme hatası: $_"
    Write-Host "gsutil komutuna erişilemiyor. gsutil.cmd'nin PATH'te olduğundan emin olun."
}

# Bucket oluşturma talimatları (gerekirse)
Write-Host "`nEğer bucket göremiyorsanız veya yeni bir bucket oluşturmanız gerekiyorsa:"
Write-Host "$gsutilPath mb gs://distinguish-ai-faces-dataset"
Write-Host "Klasör oluşturmak için:"
Write-Host "$gsutilPath cp NUL gs://distinguish-ai-faces-dataset/ai-faces/"
Write-Host "$gsutilPath cp NUL gs://distinguish-ai-faces-dataset/human-faces/"

# Ortam değişkenlerini ayarlayın
$credentialsPath = Join-Path $env:APPDATA "gcloud\application_default_credentials.json"
if (Test-Path $credentialsPath) {
    $env:GOOGLE_APPLICATION_CREDENTIALS = $credentialsPath
    Write-Host "Kimlik bilgileri dosyası bulundu ve ayarlandı: $credentialsPath"
} else {
    Write-Host "Kimlik bilgileri dosyası bulunamadı. Lütfen 'gcloud auth application-default login' komutunu çalıştırın."
}

$env:GCP_BUCKET_NAME = "distinguish-ai-faces-dataset"

Write-Host "`nGoogle Cloud konfigürasyonu tamamlandı. Şimdi training pipeline'ı çalıştırabilirsiniz."
Write-Host "Örnek komut: python main.py --model efficientnet --img-size 224 224 --batch-size 32 --epochs 20"

# Python google-cloud-storage paketini kurma (gerekirse)
Write-Host "`nGoogle Cloud Storage Python paketini kontrol ediyoruz..."
try {
    python -c "import google.cloud.storage"
    Write-Host "google-cloud-storage paketi zaten kurulu!"
} catch {
    Write-Host "google-cloud-storage paketi kuruluyor..."
    pip install google-cloud-storage
} 