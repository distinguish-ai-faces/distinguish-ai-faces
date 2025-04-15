#!/bin/bash
# AI Faces Data Pipeline çalıştırma scripti

set -e  # Herhangi bir hata olduğunda scripti durdur

# Renkli çıktı için
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}     AI Faces Data Pipeline Runner         ${NC}"
echo -e "${BLUE}============================================${NC}"

# Çalışma dizinine git
cd "$(dirname "$0")"

# Parametreler (varsayılan değerler)
TARGET_COUNT=100
BATCH_SIZE=20
RAW_DIR="./img"
PROCESSED_DIR="./img/ai_faces"
MAX_WORKERS=4
MAX_RETRIES=3
USE_GCP=true
CONTINUE_NUMBERING=true

# Parametre işleme
while [[ $# -gt 0 ]]; do
  case $1 in
    --target-count=*)
      TARGET_COUNT="${1#*=}"
      shift
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --raw-dir=*)
      RAW_DIR="${1#*=}"
      shift
      ;;
    --processed-dir=*)
      PROCESSED_DIR="${1#*=}"
      shift
      ;;
    --no-gcp)
      USE_GCP=false
      shift
      ;;
    --no-continue-numbering)
      CONTINUE_NUMBERING=false
      shift
      ;;
    --help)
      echo "Kullanım: $0 [seçenekler]"
      echo ""
      echo "Seçenekler:"
      echo "  --target-count=SAYI     Toplam indirilecek görsel sayısı (default: 100)"
      echo "  --batch-size=SAYI       Her seferde indirilecek görsel sayısı (default: 20)"
      echo "  --raw-dir=DİZİN         Ham görüntülerin kaydedileceği dizin (default: ./img)"
      echo "  --processed-dir=DİZİN   İşlenmiş görüntülerin kaydedileceği dizin (default: ./img/ai_faces)"
      echo "  --no-gcp                GCP entegrasyonunu devre dışı bırak"
      echo "  --no-continue-numbering GCP'deki mevcut dosya sayısına göre numaralandırmayı devre dışı bırak"
      echo "  --help                  Bu yardım mesajını göster"
      exit 0
      ;;
    *)
      echo "Bilinmeyen parametre: $1"
      exit 1
      ;;
  esac
done

# Virtual Environment kontrolü ve aktivasyon
if [ ! -d "venv" ]; then
  echo -e "${YELLOW}Virtual environment bulunamadı. Oluşturuluyor...${NC}"
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
else
  echo -e "${GREEN}Mevcut virtual environment kullanılıyor.${NC}"
  source venv/bin/activate
fi

# .env dosyasının varlığını kontrol et
if [ ! -f "../.env" ]; then
  echo -e "${YELLOW}Uyarı: .env dosyası bulunamadı. Varsayılan ayarlar kullanılacak.${NC}"
fi

# GCP kimlik doğrulaması
if [ "$USE_GCP" = true ]; then
  echo -e "${BLUE}GCP kimlik doğrulaması kontrol ediliyor...${NC}"
  
  # gcloud komutunun varlığını kontrol et
  if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}gcloud komutu bulunamadı. GCP işlemleri devre dışı bırakılacak.${NC}"
    echo -e "${YELLOW}GCP CLI kurulumu için: https://cloud.google.com/sdk/docs/install${NC}"
    USE_GCP=false
  else
    # Mevcut gcloud konfigürasyonunu kontrol et
    CURRENT_ACCOUNT=$(gcloud config get-value account 2>/dev/null)
    
    if [ -z "$CURRENT_ACCOUNT" ]; then
      echo -e "${YELLOW}GCP hesabı bulunamadı. Giriş yapılıyor...${NC}"
      gcloud auth login
    else
      echo -e "${GREEN}GCP hesabı: $CURRENT_ACCOUNT${NC}"
    fi
    
    # Uygulama varsayılan kimlik bilgilerini ayarla (gerekli ise)
    if [ ! -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
      echo -e "${YELLOW}Uygulama varsayılan kimlik bilgileri ayarlanıyor...${NC}"
      gcloud auth application-default login
    else
      echo -e "${GREEN}Uygulama varsayılan kimlik bilgileri bulundu.${NC}"
    fi
    
    # Aktif projeyi kontrol et
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
      echo -e "${YELLOW}Aktif GCP projesi bulunamadı.${NC}"
      read -p "Proje ID'sini girin: " PROJECT_ID
      gcloud config set project "$PROJECT_ID"
    else
      echo -e "${GREEN}Aktif GCP projesi: $PROJECT_ID${NC}"
    fi
    
    GCP_ARGS="--use-gcp"
  fi
else
  GCP_ARGS=""
  echo -e "${YELLOW}GCP entegrasyonu devre dışı bırakıldı.${NC}"
fi

# Klasörlerin varlığını kontrol et
mkdir -p "$RAW_DIR" "$PROCESSED_DIR"

# GCP'deki mevcut dosya sayısını kontrol et ve numaralandırmaya ekle
CONTINUE_NUMBERING_ARG=""
if [ "$USE_GCP" = true ] && [ "$CONTINUE_NUMBERING" = true ]; then
  echo -e "${BLUE}GCP'deki mevcut dosya sayısı kontrol ediliyor...${NC}"
  CONTINUE_NUMBERING_ARG="--continue-numbering"
fi

# Ana script'i çalıştır
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}     Data Pipeline başlatılıyor...         ${NC}"
echo -e "${BLUE}============================================${NC}"
echo -e "Hedef sayı: ${GREEN}$TARGET_COUNT${NC} görsel"
echo -e "Batch büyüklüğü: ${GREEN}$BATCH_SIZE${NC} görsel"
echo -e "GCP entegrasyonu: ${GREEN}$USE_GCP${NC}"
echo -e "Numaralandırmaya devam et: ${GREEN}$CONTINUE_NUMBERING${NC}"
echo ""

python main.py \
  --target-count="$TARGET_COUNT" \
  --batch-size="$BATCH_SIZE" \
  --raw-dir="$RAW_DIR" \
  --processed-dir="$PROCESSED_DIR" \
  --max-workers="$MAX_WORKERS" \
  --max-retries="$MAX_RETRIES" \
  $GCP_ARGS \
  $CONTINUE_NUMBERING_ARG

echo -e "${GREEN}İşlem tamamlandı!${NC}" 