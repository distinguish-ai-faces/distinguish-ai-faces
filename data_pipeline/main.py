"""
Main module for the AI face data pipeline.
Combines scraping, format conversion, and preprocessing steps.
"""
import argparse
import logging
import os
import time
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor

from src import scrape_faces
from src import preprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sabit bekleme süresi
WAIT_TIME = 10  # saniye

class AIFaceDataPipeline:
    """Main pipeline class for AI face data collection and processing."""
    
    def __init__(
        self,
        target_count: int = 1000,
        batch_size: int = 100,
        raw_dir: str = "img",
        processed_dir: str = "img/ai_faces",
        max_workers: int = 4,
        max_retries: int = 3
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
        """
        self.target_count = target_count
        self.batch_size = min(batch_size, target_count)  # Ensure batch size doesn't exceed target
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        # Ensure directories exist
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Son dosya numarasını al
        self.file_counter = self._get_last_file_number()
        logger.info(f"Starting file counter: {self.file_counter}")
        
        # Basit sayaçlar
        self.total_processed = 0
        self.current_batch = 0

    def _get_last_file_number(self) -> int:
        """Get the last file number from processed directory."""
        try:
            # Önce işlenmiş dizinindeki en yüksek sayıyı kontrol et
            files = list(Path(self.processed_dir).glob("*.jpg"))
            processed_numbers = []
            for f in files:
                try:
                    num = int(f.stem)  # "1.jpg" -> 1
                    processed_numbers.append(num)
                except ValueError:
                    continue
            
            # Sonra ham dizindeki en yüksek sayıyı kontrol et
            raw_files = list(Path(self.raw_dir).glob("*.jpg"))
            raw_numbers = []
            for f in raw_files:
                try:
                    num = int(f.stem)  # "1.jpg" -> 1
                    raw_numbers.append(num)
                except ValueError:
                    continue
            
            # En yüksek sayıyı al (ham veya işlenmiş)
            all_numbers = processed_numbers + raw_numbers
            return max(all_numbers) if all_numbers else 0
            
        except Exception as e:
            logger.error(f"Error getting last file number: {str(e)}")
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
            for i, file_path in enumerate(downloaded_files):
                file = Path(file_path)
                if not file.exists():
                    continue
                    
                new_file_number = self.file_counter + i + 1
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
                self.file_counter += len(renamed_files)
                
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
                
                # 4. Ham resimleri temizle
                self.cleanup_raw_images()
                
                # 5. İlerleme durumunu göster
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
    
    args = parser.parse_args()
    
    pipeline = AIFaceDataPipeline(
        target_count=args.target_count,
        batch_size=args.batch_size,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_workers=args.max_workers,
        max_retries=args.max_retries
    )
    
    pipeline.run_pipeline()


if __name__ == "__main__":
    main() 