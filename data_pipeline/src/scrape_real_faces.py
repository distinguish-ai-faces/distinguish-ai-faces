"""
Real human face scraper module.
Module for scraping real human faces from WhichFaceIsReal website.
"""
import asyncio
import logging
import os
import time
import random
from typing import List, Optional
import aiofiles
import aiohttp
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# Constants
DEFAULT_URL = "https://whichfaceisreal.com/index.php"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache"
}
DEFAULT_IMAGE_COUNT = 100
DEFAULT_TIMEOUT = 10  # Seconds to wait for elements to load

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealFaceScraper:
    """WhichFaceIsReal sitesinden gerçek insan yüzlerini scrape etme sınıfı."""

    def __init__(
        self,
        output_dir: str = "/Users/yucifer/Projects/distinguish-ai-faces/data_pipeline/img/human_faces/temp",
        headless: bool = False
    ):
        """
        RealFaceScraper'ı başlat.
        
        Args:
            output_dir: Görüntülerin kaydedileceği dizin
            headless: Tarayıcıyı başsız modda çalıştır
        """
        self.output_dir = output_dir
        self.url = DEFAULT_URL
        self.headless = headless
        
        # Chrome driver için seçenekleri ayarla
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument("--headless=new")  # Chrome 109+ için yeni headless modu
        self.chrome_options.add_argument("--disable-notifications")
        self.chrome_options.add_argument("--disable-infobars")
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--window-size=1920,1080")
        
        # Görüntü dizininin var olduğundan emin ol
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"RealFaceScraper başlatıldı: {self.url}")
        logger.info(f"Görüntüler şu dizine kaydedilecek: {self.output_dir}")

    def __enter__(self):
        """Context manager girişi - tarayıcıyı başlat."""
        self.driver = webdriver.Chrome(options=self.chrome_options)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager çıkışı - tarayıcıyı kapat."""
        self.driver.quit()
    
    def navigate_to_site(self):
        """WhichFaceIsReal sitesine git."""
        self.driver.get(self.url)
        logger.info(f"WhichFaceIsReal sayfasına gidiliyor: {self.url}")
        
        # Sayfanın yüklenmesi için bekle
        time.sleep(3)
    
    def click_person_one(self):
        """Person 1 butonuna tıkla."""
        try:
            # Person 1 butonunu bul ve tıkla
            # İlk resme tıkla (Person 1)
            person_one = WebDriverWait(self.driver, DEFAULT_TIMEOUT).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, ".imagecolumn img[alt='Person 1']"))
            )
            person_one.click()
            logger.info("Person 1 butonuna tıklandı")
            time.sleep(1)  # Sonucun yüklenmesi için kısa bir bekleme
            return True
        except (TimeoutException, NoSuchElementException) as e:
            logger.error(f"Person 1 butonuna tıklanamadı: {str(e)}")
            return False
    
    def find_real_image_url(self):
        """Doğru olan (gerçek insan) resmin URL'sini bul."""
        try:
            # Yeşil kenarlı (doğru) resmi bul
            real_image = WebDriverWait(self.driver, DEFAULT_TIMEOUT).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img[style*='border: 10px ridge']"))
            )
            image_url = real_image.get_attribute("src")
            logger.info(f"Gerçek yüz görüntüsü bulundu: {image_url}")
            return image_url
        except (TimeoutException, NoSuchElementException) as e:
            logger.error(f"Gerçek yüz görüntüsü bulunamadı: {str(e)}")
            return None
    
    def click_play_again(self):
        """Play Again butonuna tıkla."""
        try:
            # Play Again butonunu bul ve tıkla
            play_again = WebDriverWait(self.driver, DEFAULT_TIMEOUT).until(
                EC.element_to_be_clickable((By.XPATH, "//b[text()='Play again.']"))
            )
            play_again.click()
            logger.info("Play Again butonuna tıklandı")
            time.sleep(1)  # Yeni sayfanın yüklenmesi için kısa bir bekleme
            return True
        except (TimeoutException, NoSuchElementException) as e:
            logger.error(f"Play Again butonuna tıklanamadı: {str(e)}")
            return False
    
    async def download_image(self, session: aiohttp.ClientSession, url: str, output_path: str) -> bool:
        """
        Bir görüntüyü indir ve belirtilen dosyaya kaydet.
        
        Args:
            session: aiohttp oturumu
            url: İndirilecek görüntünün URL'si
            output_path: Görüntünün kaydedileceği dosya yolu
            
        Returns:
            İndirme başarılı ise True, aksi halde False
        """
        try:
            async with session.get(url, headers=DEFAULT_HEADERS) as response:
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if not content_type.startswith(('image/jpeg', 'image/png', 'image/webp')):
                        logger.warning(f"Görüntü değil, içerik türü: {content_type}, URL: {url}")
                        return False
                    
                    # Ham görüntü verilerini kaydet
                    image_data = await response.read()
                    async with aiofiles.open(output_path, mode='wb') as f:
                        await f.write(image_data)
                    
                    # Başarılı indirmeyi günlüğe kaydet
                    size_kb = os.path.getsize(output_path) / 1024
                    logger.info(f"Görüntü başarıyla indirildi: {output_path} ({size_kb:.1f} KB)")
                    return True
                else:
                    logger.error(f"Görüntü indirilemedi. Durum kodu: {response.status}, URL: {url}")
                    return False
        except Exception as e:
            logger.error(f"Görüntü indirilirken hata: {str(e)}, URL: {url}")
            return False
    
    def scrape_images(self, count: int = DEFAULT_IMAGE_COUNT) -> List[str]:
        """
        WhichFaceIsReal sitesinden gerçek insan yüzlerini scrape et.
        
        Args:
            count: İndirilecek görüntü sayısı
            
        Returns:
            İndirilen görüntü dosyalarının yolları
        """
        downloaded_files = []
        
        # Siteye git
        self.navigate_to_site()
        
        # Belirtilen sayıda görüntü için döngü
        for i in range(count):
            try:
                logger.info(f"Görüntü {i+1}/{count} için işlem başlatılıyor")
                
                # Person 1'e tıkla
                if not self.click_person_one():
                    continue
                
                # Gerçek resmin URL'sini al
                image_url = self.find_real_image_url()
                if not image_url:
                    continue
                
                # Görüntüyü indir
                file_path = os.path.join(self.output_dir, f"{i+1}.jpg")
                response = requests.get(image_url, headers=DEFAULT_HEADERS)
                
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    # İndirilen dosyayı listeye ekle
                    downloaded_files.append(file_path)
                    logger.info(f"Görüntü {i+1}/{count} indirildi: {file_path}")
                else:
                    logger.error(f"Görüntü {i+1} indirilemedi. Durum kodu: {response.status_code}")
                
                # Play Again'e tıkla
                if not self.click_play_again():
                    continue
                
            except Exception as e:
                logger.error(f"Görüntü {i+1} scrape edilirken hata: {str(e)}")
                continue
        
        logger.info(f"Toplam {len(downloaded_files)}/{count} görüntü başarıyla indirildi")
        return downloaded_files


def run_real_face_scraper(
    output_dir: str = "/Users/yucifer/Projects/distinguish-ai-faces/data_pipeline/img/human_faces/temp",
    count: int = DEFAULT_IMAGE_COUNT,
    headless: bool = False
) -> List[str]:
    """
    WhichFaceIsReal sitesinden gerçek insan yüzlerini scrape et.
    
    Args:
        output_dir: Görüntülerin kaydedileceği dizin
        count: İndirilecek görüntü sayısı
        headless: Tarayıcıyı başsız modda çalıştır
        
    Returns:
        İndirilen görüntü dosyalarının yolları
    """
    with RealFaceScraper(output_dir=output_dir, headless=headless) as scraper:
        return scraper.scrape_images(count=count)


if __name__ == "__main__":
    # Script doğrudan çalıştırıldığında 10 görüntü indirmeyi dene
    run_real_face_scraper(count=10, headless=False) 