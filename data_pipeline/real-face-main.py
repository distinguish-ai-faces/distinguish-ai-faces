"""
Main module for the real human face data pipeline.
Combines scraping from WhichFaceIsReal and preprocessing steps.
"""
import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Import yollarını düzelt
from src import scrape_real_faces
from src import preprocess
from src import gcp_storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sabit bekleme süresi
WAIT_TIME = 10  # saniye

# GCP bucket ve klasörler
GCP_BUCKET = os.getenv("GCP_BUCKET_NAME", "distinguish-ai-faces-dataset")
GCP_HUMAN_FACES_FOLDER = "human-faces"

class RealFaceDataPipeline:
    """Main pipeline class for real human face data collection and processing."""
    
    def __init__(
        self,
        target_count: int = 100,
        batch_size: int = 10,
        raw_dir: str = "/Users/yucifer/Projects/distinguish-ai-faces/data_pipeline/img/human_faces/temp",
        processed_dir: str = "/Users/yucifer/Projects/distinguish-ai-faces/data_pipeline/img/human_faces",
        max_workers: int = 4,
        max_retries: int = 3,
        headless: bool = False,
        use_gcp: bool = True,
        gcp_bucket: str = GCP_BUCKET,
        continue_numbering: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            target_count: Total number of images to collect
            batch_size: Number of images to process in each batch
            raw_dir: Directory for raw downloaded images
            processed_dir: Directory for processed images
            max_workers: Maximum number of worker threads
            max_retries: Maximum number of retry attempts
            headless: Whether to run browser in headless mode
            use_gcp: Whether to upload processed images to GCP
            gcp_bucket: GCP bucket name
            continue_numbering: Whether to continue numbering from last file
        """
        logger.info("Pipeline başlatılıyor...")
        
        self.target_count = target_count
        self.batch_size = min(batch_size, target_count)
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.headless = headless
        self.use_gcp = use_gcp
        self.gcp_bucket = gcp_bucket
        self.continue_numbering = continue_numbering
        self.gcp_file_count = 0
        
        logger.info(f"Hedef sayı: {target_count}")
        logger.info(f"Batch büyüklüğü: {batch_size}")
        logger.info(f"Ham görüntü dizini: {raw_dir}")
        logger.info(f"İşlenmiş görüntü dizini: {processed_dir}")
        logger.info(f"Headless mod: {headless}")
        logger.info(f"GCP kullanımı: {use_gcp}")
        logger.info(f"GCP bucket: {gcp_bucket}")
        logger.info(f"Numaralandırmaya devam et: {continue_numbering}")
        
        # Ensure directories exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Bucket erişimini kontrol et ve GCP'deki dosya sayısını al
        if self.use_gcp:
            logger.info("GCP bucket erişimi kontrol ediliyor...")
            if not gcp_storage.validate_bucket_access(self.gcp_bucket):
                logger.warning(f"GCP bucket '{self.gcp_bucket}' erişilemiyor. GCP upload devre dışı bırakıldı.")
                self.use_gcp = False
            else:
                logger.info("GCP bucket erişimi başarılı.")
                # GCP'deki dosya sayısını hemen al
                self.gcp_file_count = self._get_gcp_file_count()
                logger.info(f"GCP'de mevcut {self.gcp_file_count} dosya bulundu.")
                logger.info(f"Yeni dosyalar {self.gcp_file_count + 1}'den başlayacak.")
        
        # Dosya sayacını belirle (ya GCP'deki ya da yerel dizindeki son dosya numarasından)
        local_file_count = self._get_last_file_number(processed_dir)
        self.file_counter = max(self.gcp_file_count, local_file_count)
        logger.info(f"Başlangıç dosya sayacı: {self.file_counter}")
        
        # Basit sayaçlar
        self.total_processed = 0
        self.current_batch = 0

    def _get_gcp_file_count(self) -> int:
        """Get the total number of files in GCP bucket."""
        try:
            logger.info("GCP'den dosya sayısı alınıyor...")
            
            # GCP bucket'daki dosyaları listele
            files = gcp_storage.list_bucket_files(
                bucket_name=self.gcp_bucket,
                prefix=f"{GCP_HUMAN_FACES_FOLDER}/"
            )
            
            # Debug: Tüm dosyaları logla
            logger.info("GCP'den gelen dosya listesi:")
            file_list = list(files)  # Iterator'ı listeye çevir
            for file in file_list:
                logger.info(f"Dosya: {file.get('name', 'N/A')}")
            
            # Sadece .jpg dosyalarını say
            file_count = sum(1 for file in file_list if file["name"].lower().endswith('.jpg'))
            
            logger.info(f"GCP bucket '{self.gcp_bucket}' içinde toplam {len(file_list)} dosya bulundu")
            logger.info(f"Bunlardan {file_count} tanesi .jpg uzantılı")
            
            if file_count == 0:
                logger.warning("GCP'de hiç .jpg dosyası bulunamadı!")
            
            return file_count
            
        except Exception as e:
            logger.error(f"GCP'den dosya sayısı alınırken hata oluştu: {str(e)}")
            logger.error(f"Hata detayı: {type(e).__name__}")
            return 0

    def _get_last_file_number(self, directory: str) -> int:
        """
        Dizindeki en son dosya numarasını bul.
        
        Args:
            directory: Dosyaların bulunduğu dizin
            
        Returns:
            En son dosya numarası (0 ise dosya yoktur)
        """
        try:
            files = list(Path(directory).glob("*.jpg"))
            if not files:
                return 0
                
            # Dosya adlarını sayıya çevir, sayı olmayanları filtrele
            numbers = []
            for file in files:
                try:
                    numbers.append(int(file.stem))
                except ValueError:
                    pass
            
            if not numbers:
                return 0
                
            return max(numbers)
            
        except Exception as e:
            logger.error(f"Dosya numarası belirlenirken hata oluştu: {str(e)}")
            return 0

    def download_images(self, count: int) -> List[str]:
        """
        WhichFaceIsReal sitesinden gerçek insan yüzlerini indir.
        
        Args:
            count: İndirilecek görüntü sayısı
            
        Returns:
            İndirilen dosya yollarının listesi
        """
        try:
            logger.info(f"{count} adet gerçek insan yüzü indiriliyor...")
            
            # WhichFaceIsReal'den indir
            downloaded_files = scrape_real_faces.run_real_face_scraper(
                output_dir=self.raw_dir,
                count=count,
                headless=self.headless
            )
            
            if downloaded_files:
                logger.info(f"Toplam {len(downloaded_files)} adet görüntü indirildi")
            else:
                logger.warning("Hiç görüntü indirilemedi!")
                
            return downloaded_files
            
        except Exception as e:
            logger.error(f"İndirme sırasında hata oluştu: {str(e)}")
            return []

    def rename_files_sequentially(self, files: List[str]) -> List[str]:
        """
        İndirilen dosyaları sıralı olarak yeniden adlandır.
        
        Args:
            files: Yeniden adlandırılacak dosyaların yollarının listesi
            
        Returns:
            Yeniden adlandırılmış dosya yollarının listesi
        """
        if not files:
            return []
            
        renamed_files = []
        start_number = self.file_counter + 1
        
        for i, file_path in enumerate(files):
            file = Path(file_path)
            if not file.exists():
                continue
                
            new_file_number = start_number + i
            # Temp dosyada aynı isimle (1.jpg, 2.jpg, ...) ancak farklı içerikle dosyalar var
            # Bu nedenle, işlenmiş dizindeki dosya adı ile çakışma olmayacak
            # Rename işlemi yapmıyoruz, sadece dosya yolunu döndürüyoruz
            renamed_files.append(str(file_path))
            logger.debug(f"Dosya: {file_path} → İşlendikten sonra: {new_file_number}.jpg olarak kaydedilecek")
        
        # Dosya sayacını güncelle
        if renamed_files:
            self.file_counter = start_number + len(renamed_files) - 1
            logger.info(f"Yeni dosya numaraları: {start_number} - {self.file_counter}")
            
        return renamed_files

    def process_images(self, input_files: List[str]) -> int:
        """
        İndirilen görüntüleri işle.
        
        Args:
            input_files: İşlenecek dosyaların yollarının listesi
            
        Returns:
            Başarıyla işlenen görüntü sayısı
        """
        if not input_files:
            logger.warning("İşlenecek görüntü yok!")
            return 0
        
        start_number = self.file_counter - len(input_files) + 1
        
        # Raw dosyaları işle ve sıralı isimlerle kaydet
        successful_files = []
        
        for i, file_path in enumerate(input_files):
            input_path = Path(file_path)
            if not input_path.exists():
                logger.warning(f"Dosya bulunamadı: {file_path}")
                continue
                
            # Sıralı isimle çıktı dosyası oluştur
            output_filename = f"{start_number + i}.jpg"
            output_path = Path(self.processed_dir) / output_filename
            
            # Görüntü işleme adımları
            try:
                # İşleme için görüntüyü yükle
                img = preprocess.resize_and_crop_image(
                    str(input_path),
                    output_size=(512, 512),
                    center_face=True
                )
                
                # İşlenmiş görüntüyü kaydet
                from PIL import Image
                processed_pil = Image.fromarray(img)
                processed_pil.save(output_path, format='JPEG', quality=95)
                
                successful_files.append(str(output_path))
                logger.info(f"Başarıyla işlendi: {input_path} → {output_path}")
                
            except Exception as e:
                logger.error(f"Görüntü işlenirken hata oluştu {input_path}: {str(e)}")
        
        # İşleme sonuçlarını kaydet
        success_count = len(successful_files)
        logger.info(f"Toplam {success_count}/{len(input_files)} görüntü başarıyla işlendi")
        
        return success_count

    def upload_processed_images_to_gcp(self) -> int:
        """
        Upload processed images to GCP bucket.
        
        Returns:
            Number of successfully uploaded files
        """
        if not self.use_gcp:
            logger.info("GCP upload devre dışı olduğu için atlanıyor.")
            return 0
        
        try:
            # İşlenmiş dizindeki tüm görüntüleri bul
            image_files = list(Path(self.processed_dir).glob("*.jpg"))
            
            # Yüklenen dosyaların listesini tut
            uploaded_files = []
            upload_results = []
            
            # Tüm görseller için
            for img_path in image_files:
                try:
                    # Dosya adını al (1.jpg, 2.jpg, vb.)
                    file_name = img_path.name
                    
                    # GCP hedef yolunu oluştur
                    gcp_path = f"{GCP_HUMAN_FACES_FOLDER}/{file_name}"
                    
                    # Dosyayı yükle
                    result = gcp_storage.upload_file(
                        local_file=img_path,
                        bucket_name=self.gcp_bucket,
                        destination_path=gcp_path
                    )
                    
                    upload_results.append(result)
                    
                    if result["success"]:
                        uploaded_files.append(str(img_path))
                        logger.debug(f"Başarıyla yüklendi: {img_path} → {gcp_path}")
                    else:
                        logger.error(f"Yükleme başarısız {img_path}: {result.get('error', 'Bilinmeyen hata')}")
                
                except Exception as e:
                    logger.error(f"Yükleme sırasında hata oluştu {img_path}: {str(e)}")
            
            # Özet
            success_count = sum(1 for r in upload_results if r.get("success", False))
            logger.info(f"GCP bucket '{self.gcp_bucket}' içine {success_count}/{len(image_files)} görüntü yüklendi")
            
            return success_count
            
        except Exception as e:
            logger.error(f"İşlenmiş görüntüler GCP'ye yüklenirken hata oluştu: {str(e)}")
            return 0

    def cleanup_local_files(self, processed_only: bool = False):
        """
        Clean up local image files.
        
        Args:
            processed_only: If True, only removes processed images, otherwise removes both raw and processed images
        """
        # Processed images
        try:
            image_files = list(Path(self.processed_dir).glob("*.jpg"))
            for file_path in image_files:
                try:
                    os.remove(file_path)
                    logger.debug(f"İşlenmiş görüntü silindi: {file_path}")
                except OSError as e:
                    logger.error(f"Dosya silinirken hata oluştu {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"İşlenmiş görüntüler temizlenirken hata oluştu: {str(e)}")
        
        # Raw images (only if processed_only is False)
        if not processed_only:
            self.cleanup_raw_images()

    def cleanup_raw_images(self):
        """Ham görüntüleri işlemeden sonra temizle."""
        try:
            image_files = list(Path(self.raw_dir).glob("*.jpg"))
            for file_path in image_files:
                try:
                    os.remove(file_path)
                    logger.debug(f"Ham görüntü silindi: {file_path}")
                except OSError as e:
                    logger.error(f"Dosya silinirken hata oluştu {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Ham görüntüler temizlenirken hata oluştu: {str(e)}")

    def print_progress(self):
        """İlerleme durumunu göster."""
        progress_percent = (self.total_processed / self.target_count) * 100 if self.target_count > 0 else 0
        logger.info(f"""
        İlerleme Durumu:
        - Hedef: {self.target_count} görüntü
        - İşlenen: {self.total_processed} görüntü ({progress_percent:.1f}%)
        - Mevcut batch: {self.current_batch}
        - Sonraki görüntü numarası: {self.file_counter + 1}
        """)

    def run_pipeline(self):
        """Hedef sayıya ulaşana kadar pipeline'ı çalıştır."""
        logger.info(f"Pipeline başlatılıyor. Hedef: {self.target_count} görüntü")
        
        # Validate GCP connection if using GCP
        if self.use_gcp:
            logger.info(f"GCP entegrasyonu etkin. Bucket: {self.gcp_bucket}")
            if gcp_storage.validate_bucket_access(self.gcp_bucket):
                logger.info("GCP bucket erişimi başarıyla doğrulandı.")
            else:
                logger.warning(f"GCP bucket'a erişilemiyor. Yüklemeler atlanacak.")
                self.use_gcp = False
        
        while self.total_processed < self.target_count:
            self.current_batch += 1
            retry_count = 0
            
            try:
                # Kalan resim sayısını hesapla
                remaining = self.target_count - self.total_processed
                current_batch_size = min(self.batch_size, remaining)
                
                if current_batch_size <= 0:
                    break
                
                logger.info(f"Batch {self.current_batch}: {current_batch_size} görüntü indiriliyor...")
                
                # 1. Resimleri indir
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(self.download_images, current_batch_size)
                    downloaded = future.result()
                
                logger.info(f"Batch {self.current_batch}: {len(downloaded)} görüntü indirildi")
                
                # 2. Dosyaları sıralı olarak yeniden adlandır
                renamed_files = self.rename_files_sequentially(downloaded)
                
                # Sabit bekleme süresi
                logger.info(f"{WAIT_TIME} saniye bekleniyor...")
                time.sleep(WAIT_TIME)
                
                # 3. Resimleri işle
                logger.info(f"Batch {self.current_batch}: Görüntüler işleniyor...")
                success_count = self.process_images(renamed_files)
                
                # 4. İşlenen resim sayısını güncelle
                self.total_processed += success_count
                
                # 5. İşlenmiş resimleri GCP'ye yükle
                if self.use_gcp and success_count > 0:
                    logger.info(f"Batch {self.current_batch}: İşlenmiş görüntüler GCP'ye yükleniyor...")
                    upload_count = self.upload_processed_images_to_gcp()
                    logger.info(f"Batch {self.current_batch}: GCP'ye {upload_count} görüntü yüklendi")
                
                # 6. Ham ve işlenmiş resimleri temizle (GCP'ye yükledikten sonra)
                if self.use_gcp:
                    logger.info(f"Batch {self.current_batch}: Yerel dosyalar temizleniyor...")
                    self.cleanup_local_files(processed_only=False)
                else:
                    # GCP kullanılmıyorsa sadece ham dosyaları temizle
                    self.cleanup_raw_images()
                
                # 7. İlerleme durumunu göster
                self.print_progress()
                
                # Hedef sayıya ulaşıldı mı kontrol et
                if self.total_processed >= self.target_count:
                    logger.info("Hedef sayıya ulaşıldı!")
                    break
                
                # Sabit bekleme süresi
                logger.info(f"Sonraki batch için {WAIT_TIME} saniye bekleniyor...")
                time.sleep(WAIT_TIME)
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Batch {self.current_batch} sırasında hata oluştu: {str(e)}")
                
                if retry_count < self.max_retries:
                    logger.warning(f"Batch başarısız oldu, yeniden deneniyor... ({retry_count}/{self.max_retries})")
                    time.sleep(WAIT_TIME)
                    continue
                else:
                    logger.error(f"Batch {self.max_retries} deneme sonrasında başarısız oldu.")
                    raise
        
        logger.info(f"""
        Pipeline tamamlandı!
        - Toplam işlenen görüntü: {self.total_processed}
        - Toplam batch: {self.current_batch}
        """)


def main():
    """Komut satırı argümanlarını ayrıştır ve pipeline'ı çalıştır."""
    parser = argparse.ArgumentParser(description="Gerçek Yüz Veri Pipeline'ı")
    parser.add_argument("--target-count", type=int, required=True, help="Toplanacak toplam görüntü sayısı")
    parser.add_argument("--batch-size", type=int, default=10, help="Her batch için görüntü sayısı")
    parser.add_argument("--raw-dir", type=str, default="/Users/yucifer/Projects/distinguish-ai-faces/data_pipeline/img/human_faces/temp", help="Ham görüntüler için dizin")
    parser.add_argument("--processed-dir", type=str, default="/Users/yucifer/Projects/distinguish-ai-faces/data_pipeline/img/human_faces", help="İşlenmiş görüntüler için dizin")
    parser.add_argument("--max-workers", type=int, default=4, help="Maksimum worker thread sayısı")
    parser.add_argument("--max-retries", type=int, default=3, help="Maksimum yeniden deneme sayısı")
    parser.add_argument("--headless", action="store_true", help="Tarayıcıyı başsız modda çalıştır")
    parser.add_argument("--use-gcp", action="store_true", help="İşlenmiş görüntüleri GCP bucket'a yükle")
    parser.add_argument("--gcp-bucket", type=str, default=GCP_BUCKET, help="GCP bucket adı")
    parser.add_argument("--continue-numbering", action="store_true", help="GCP'deki son dosyadan numaralandırmaya devam et")
    
    args = parser.parse_args()
    
    pipeline = RealFaceDataPipeline(
        target_count=args.target_count,
        batch_size=args.batch_size,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        headless=args.headless,
        use_gcp=args.use_gcp,
        gcp_bucket=args.gcp_bucket,
        continue_numbering=args.continue_numbering
    )
    
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 