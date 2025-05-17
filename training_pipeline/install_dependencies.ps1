# Python bağımlılıklarını kuran PowerShell betiği

Write-Host "Python bağımlılıkları kuruluyor... Bu işlem biraz zaman alabilir."

# Temel bağımlılıkları kur
Write-Host "`nTemel bağımlılıklar kuruluyor..."
pip install -r requirements.txt

# Özellikle Google Cloud Storage'ı kontrol et
Write-Host "`nGoogle Cloud Storage paketini kontrol ediyoruz..."
try {
    python -c "import google.cloud.storage; print('Google Cloud Storage paketi başarıyla kuruldu!')"
} catch {
    Write-Host "Google Cloud Storage paketi eksik, tekrar kuruluyor..."
    pip install google-cloud-storage
}

# Tensorflow'u kontrol et
Write-Host "`nTensorFlow paketini kontrol ediyoruz..."
try {
    python -c "import tensorflow as tf; print(f'TensorFlow versiyonu: {tf.__version__}')"
} catch {
    Write-Host "TensorFlow paketi eksik, tekrar kuruluyor..."
    pip install tensorflow
}

Write-Host "`nBağımlılık kurulumu tamamlandı."
Write-Host "Şimdi GCP yapılandırmasını yapmak için aşağıdaki komutu çalıştırabilirsiniz:"
Write-Host "./setup_gcp.ps1" 