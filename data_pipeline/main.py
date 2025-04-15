"""
Main module for the AI face data pipeline.
Combines scraping, format conversion, and preprocessing steps.
"""
import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from src import scrape_faces
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
GCP_AI_FACES_FOLDER = "ai-faces"
GCP_HUMAN_FACES_FOLDER = "human-faces"  # Henüz kullanılmıyor ama ileride kullanılabilir

class AIFaceDataPipeline:
    """Main pipeline class for AI face data collection and processing."""
    
    def __init__(
        self,
        target_count: int = 1000,
        batch_size: int = 100,
        raw_dir: str = "img",
        processed_dir: str = "img/ai_faces",
        max_workers: int = 4,
        max_retries: int = 3,
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
        self.use_gcp = use_gcp
        self.gcp_bucket = gcp_bucket
        self.continue_numbering = continue_numbering
        self.gcp_file_count = 0
        
        logger.info(f"Hedef sayı: {target_count}")
        logger.info(f"Batch büyüklüğü: {batch_size}")
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
        
        # Dosya sayacını GCP'deki dosya sayısından başlat
        self.file_counter = self.gcp_file_count
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
                prefix=f"{GCP_AI_FACES_FOLDER}/"
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

    def download_images(self, count: int) -> List[str]:
        """Download images and rename them sequentially."""
        try:
            # Konfigürasyonu hazırla
            config = {
                'url': scrape_faces.DEFAULT_URL,
                'rate_limit': 1.0,
                'requires_js': False
            }
            
            # Scraper'ı oluştur
            scraper = scrape_faces.ImageScraper(config=config, img_dir=self.raw_dir)
            
            # Asenkron indirme işlemini çalıştır
            import asyncio
            downloaded_files = asyncio.run(scraper.download_multiple_images(count, "temp"))
            
            # İndirilen dosyaları sıralı olarak yeniden adlandır
            renamed_files = []
            start_number = self.file_counter + 1  # GCP'deki son dosya numarasından başla
            
            for i, file_path in enumerate(downloaded_files):
                file = Path(file_path)
                if not file.exists():
                    continue
                    
                new_file_number = start_number + i
                new_file_path = file.parent / f"{new_file_number}.jpg"
                
                # Eğer hedef dosya zaten varsa silmeyi dene
                if new_file_path.exists():
                    try:
                        os.remove(new_file_path)
                    except Exception as e:
                        logger.error(f"Could not remove existing file {new_file_path}: {str(e)}")
                        continue
                
                try:
                    os.rename(file_path, new_file_path)
                    renamed_files.append(str(new_file_path))
                    logger.debug(f"Renamed {file_path} to {new_file_path}")
                except Exception as e:
                    logger.error(f"Error renaming {file_path}: {str(e)}")
            
            # Dosya sayacını güncelle
            if renamed_files:
                self.file_counter = start_number + len(renamed_files) - 1
                logger.info(f"Yeni dosya numaraları: {start_number} - {self.file_counter}")
                
            return renamed_files
            
        except Exception as e:
            logger.error(f"Error in download_images: {str(e)}")
            return []

    def process_images(self) -> int:
        """Process all images in the raw directory."""
        results = preprocess.process_scraped_images(
            input_dir=self.raw_dir,
            output_dir=self.processed_dir,
            output_size=(512, 512),
            center_face=True,
            remove_watermark=True,
            target_format="jpg",
            quality=95
        )
        
        # Başarılı işlem sayısını say
        success_count = sum(1 for r in results if r.get('success', False))
        logger.info(f"Processed {success_count}/{len(results)} images successfully")
        
        return success_count

    def upload_processed_images_to_gcp(self) -> int:
        """
        Upload processed images to GCP bucket.
        
        Returns:
            Number of successfully uploaded files
        """
        if not self.use_gcp:
            logger.info("GCP upload skipped as use_gcp is disabled.")
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
                    gcp_path = f"{GCP_AI_FACES_FOLDER}/{file_name}"
                    
                    # Dosyayı yükle
                    result = gcp_storage.upload_file(
                        local_file=img_path,
                        bucket_name=self.gcp_bucket,
                        destination_path=gcp_path
                    )
                    
                    upload_results.append(result)
                    
                    if result["success"]:
                        uploaded_files.append(str(img_path))
                        logger.debug(f"Successfully uploaded {img_path} to {gcp_path}")
                    else:
                        logger.error(f"Failed to upload {img_path}: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    logger.error(f"Error uploading {img_path}: {str(e)}")
            
            # Özet
            success_count = sum(1 for r in upload_results if r.get("success", False))
            logger.info(f"Uploaded {success_count}/{len(image_files)} images to GCP bucket '{self.gcp_bucket}'")
            
            return success_count
            
        except Exception as e:
            logger.error(f"Error uploading processed images to GCP: {str(e)}")
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
                    logger.debug(f"Removed processed image: {file_path}")
                except OSError as e:
                    logger.error(f"Error removing {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error cleaning up processed images: {str(e)}")
        
        # Raw images (only if processed_only is False)
        if not processed_only:
            self.cleanup_raw_images()

    def cleanup_raw_images(self):
        """Clean up original images after processing."""
        try:
            image_files = list(Path(self.raw_dir).glob("*.jpg"))
            for file_path in image_files:
                try:
                    os.remove(file_path)
                    logger.debug(f"Removed original image: {file_path}")
                except OSError as e:
                    logger.error(f"Error removing {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error cleaning up raw images: {str(e)}")

    def print_progress(self):
        """Print current progress."""
        progress_percent = (self.total_processed / self.target_count) * 100 if self.target_count > 0 else 0
        logger.info(f"""
        Progress Update:
        - Target: {self.target_count} images
        - Processed: {self.total_processed} images ({progress_percent:.1f}%)
        - Current Batch: {self.current_batch}
        - Next image number: {self.file_counter + 1}
        """)

    def run_pipeline(self):
        """Run the pipeline until target count is reached."""
        logger.info(f"Starting pipeline. Target: {self.target_count} images")
        
        # Validate GCP connection if using GCP
        if self.use_gcp:
            logger.info(f"GCP integration enabled. Using bucket: {self.gcp_bucket}")
            if gcp_storage.validate_bucket_access(self.gcp_bucket):
                logger.info("GCP bucket access validated successfully.")
            else:
                logger.warning(f"Could not access GCP bucket. Uploads will be skipped.")
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
                
                logger.info(f"Batch {self.current_batch}: Downloading {current_batch_size} images...")
                
                # 1. Resimleri indir
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future = executor.submit(self.download_images, current_batch_size)
                    downloaded = future.result()
                
                logger.info(f"Batch {self.current_batch}: Downloaded {len(downloaded)} images")
                
                # Sabit bekleme süresi
                logger.info(f"Waiting {WAIT_TIME} seconds before processing...")
                time.sleep(WAIT_TIME)
                
                # 2. Resimleri işle
                logger.info(f"Batch {self.current_batch}: Processing images...")
                success_count = self.process_images()
                
                # 3. İşlenen resim sayısını güncelle
                self.total_processed += success_count
                
                # 4. İşlenmiş resimleri GCP'ye yükle
                if self.use_gcp and success_count > 0:
                    logger.info(f"Batch {self.current_batch}: Uploading processed images to GCP...")
                    upload_count = self.upload_processed_images_to_gcp()
                    logger.info(f"Batch {self.current_batch}: Uploaded {upload_count} images to GCP")
                
                # 5. Ham ve işlenmiş resimleri temizle (GCP'ye yükledikten sonra)
                if self.use_gcp:
                    logger.info(f"Batch {self.current_batch}: Cleaning up local files...")
                    self.cleanup_local_files(processed_only=False)
                else:
                    # GCP kullanılmıyorsa sadece ham dosyaları temizle
                    self.cleanup_raw_images()
                
                # 6. İlerleme durumunu göster
                self.print_progress()
                
                # Hedef sayıya ulaşıldı mı kontrol et
                if self.total_processed >= self.target_count:
                    logger.info("Target count reached!")
                    break
                
                # Sabit bekleme süresi
                logger.info(f"Waiting {WAIT_TIME} seconds before next batch...")
                time.sleep(WAIT_TIME)
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Error in batch {self.current_batch}: {str(e)}")
                
                if retry_count < self.max_retries:
                    logger.warning(f"Batch failed, retrying... ({retry_count}/{self.max_retries})")
                    time.sleep(WAIT_TIME)
                    continue
                else:
                    logger.error(f"Batch failed after {self.max_retries} retries.")
                    raise
        
        logger.info(f"""
        Pipeline completed!
        - Total images processed: {self.total_processed}
        - Total batches: {self.current_batch}
        """)


def main():
    """Parse command-line arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="AI Face Data Pipeline")
    parser.add_argument("--target-count", type=int, required=True, help="Total number of images to collect")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of images per batch")
    parser.add_argument("--raw-dir", type=str, default="img", help="Directory for raw images")
    parser.add_argument("--processed-dir", type=str, default="img/ai_faces", help="Directory for processed images")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retry attempts")
    parser.add_argument("--use-gcp", action="store_true", help="Upload processed images to GCP bucket")
    parser.add_argument("--gcp-bucket", type=str, default=GCP_BUCKET, help="GCP bucket name")
    parser.add_argument("--continue-numbering", action="store_true", help="Continue numbering from last file in GCP")
    
    args = parser.parse_args()
    
    pipeline = AIFaceDataPipeline(
        target_count=args.target_count,
        batch_size=args.batch_size,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        use_gcp=args.use_gcp,
        gcp_bucket=args.gcp_bucket,
        continue_numbering=args.continue_numbering
    )
    
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 