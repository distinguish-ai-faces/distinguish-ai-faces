"""
AI-generated face scraper module.
Module for scraping AI-generated faces from various sources.
"""
import asyncio
import logging
import os
import time
import random
from typing import Dict, List, Optional, Tuple, Union
import urllib.robotparser

import aiofiles
import aiohttp
import requests
from tqdm import tqdm

# Constants
DEFAULT_URL = "https://thispersondoesnotexist.com"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache"
}
DEFAULT_DELAY = 1.0  # Base delay in seconds between requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_connection(url: str, timeout: int = 10) -> bool:
    """Check if the URL is accessible."""
    response = requests.get(url, timeout=timeout)  # Let the exception propagate
    return response.status_code == 200


def validate_image(response: requests.Response) -> bool:
    """Validate if the response contains a valid image."""
    content_type = response.headers.get('Content-Type', '')
    return content_type.startswith(('image/jpeg', 'image/png', 'image/webp'))


def download_image(url: str, headers: Dict[str, str] = None, proxies: Dict[str, str] = None) -> Optional[bytes]:
    """Download an image from a URL."""
    try:
        response = requests.get(url, headers=headers or DEFAULT_HEADERS, proxies=proxies)
        if response.status_code == 200 and validate_image(response):
            return response.content
        return None
    except requests.RequestException:
        return None


class ImageScraper:
    """Class for scraping AI-generated faces."""

    def __init__(
        self,
        config: Dict,
        img_dir: str = "img",
    ):
        """
        Initialize ImageScraper.

        Args:
            config: Site configuration dictionary
            img_dir: Directory to save images to
        """
        self.url = config['url']
        self.rate_limit = config.get('rate_limit', DEFAULT_DELAY)
        self.image_selector = config.get('image_selector', 'img')
        self.requires_js = config.get('requires_js', False)
        self.headers = DEFAULT_HEADERS.copy()
        self.img_dir = img_dir

        # Ensure the image directory exists
        os.makedirs(self.img_dir, exist_ok=True)
        logger.info(f"Initialized ImageScraper for {self.url}")

    def check_rate_limit(self) -> bool:
        """Check if we're being rate limited."""
        try:
            response = requests.get(self.url, headers=self.headers)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def check_robots_txt(self) -> bool:
        """Check if scraping is allowed by robots.txt."""
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{self.url}/robots.txt")
        try:
            rp.read()
            return rp.can_fetch(self.headers['User-Agent'], self.url)
        except:
            return True  # Assume allowed if robots.txt is inaccessible

    def get_page_content(self) -> Optional[str]:
        """Get page content, handling JavaScript if required."""
        if self.requires_js:
            try:
                from selenium import webdriver
                from selenium.webdriver.chrome.options import Options
                
                options = Options()
                options.add_argument('--headless')
                driver = webdriver.Chrome(options=options)
                driver.get(self.url)
                content = driver.page_source
                driver.quit()
                return content
            except Exception as e:
                logger.error(f"Selenium error: {e}")
                return None
        else:
            try:
                response = requests.get(self.url, headers=self.headers)
                return response.text if response.status_code == 200 else "<html></html>"  # Return empty HTML instead of None
            except requests.RequestException:
                return "<html></html>"  # Return empty HTML instead of None

    def download_image(self, url: str, proxies: Dict[str, str] = None) -> Optional[bytes]:
        """Download a single image."""
        return download_image(url, self.headers, proxies)

    def download_image_with_retry(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download image with retry mechanism."""
        for attempt in range(max_retries):
            image_data = self.download_image(url)
            if image_data:
                return image_data
            time.sleep(self.rate_limit * (attempt + 1))
        return None

    def download_batch(self, urls: List[str], max_workers: int = 3) -> List[bytes]:
        """Download multiple images in parallel."""
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(
                lambda url: self.download_image_with_retry(url),
                urls
            ))

    async def download_single_image(
        self,
        session: aiohttp.ClientSession,
        index: int,
        filename_prefix: str = "ai-face"
    ) -> Tuple[bool, Optional[str]]:
        """
        Download a single AI-generated face image.

        Args:
            session: aiohttp client session
            index: Image index (for naming and delay calculation)
            filename_prefix: Prefix for the image filename

        Returns:
            Tuple of (success boolean, file path or None if failed)
        """
        try:
            # Add random delay to avoid getting cached images
            delay = self.rate_limit + random.uniform(0.5, 1.5)
            await asyncio.sleep(delay * (index + 0.1))
            
            # Add timestamp to headers to prevent caching
            headers = self.headers.copy()
            headers["X-Request-Time"] = str(time.time())
            
            async with session.get(self.url, headers=headers) as response:
                if response.status == 200:
                    # Generate filename
                    file_path = os.path.join(self.img_dir, f"{filename_prefix}-{index+1}.jpg")
                    
                    # Save the raw image data
                    image_data = await response.read()
                    async with aiofiles.open(file_path, mode='wb') as f:
                        await f.write(image_data)
                    
                    # Log successful download
                    size_kb = os.path.getsize(file_path) / 1024
                    logger.info(f"Image {index+1} downloaded successfully ({size_kb:.1f} KB)")
                    return True, file_path
                else:
                    logger.error(f"Failed to download image {index+1}. Status code: {response.status}")
                    return False, None
        except Exception as e:
            logger.error(f"Error downloading image {index+1}: {str(e)}")
            return False, None

    async def download_multiple_images(
        self,
        count: int,
        filename_prefix: str = "ai-face"
    ) -> List[str]:
        """
        Download multiple AI-generated face images.

        Args:
            count: Number of images to download
            filename_prefix: Prefix for the image filenames

        Returns:
            List of successfully downloaded image file paths
        """
        downloaded_files = []
        start_time = time.time()

        # Initialize progress bar
        pbar = tqdm(total=count, desc="Downloading images")

        # Create and reuse a single aiohttp session for all requests
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(count):
                task = self.download_single_image(session, i, filename_prefix)
                tasks.append(task)
            
            # Process downloads as they complete
            for future in asyncio.as_completed(tasks):
                success, file_path = await future
                pbar.update(1)
                if success and file_path:
                    downloaded_files.append(file_path)
        
        pbar.close()
        
        # Report results
        elapsed = time.time() - start_time
        logger.info(f"Downloaded {len(downloaded_files)}/{count} images in {elapsed:.1f} seconds")
        return downloaded_files


async def download_faces(
    count: int = 5,
    output_dir: str = "img",
    filename_prefix: str = "ai-face",
    delay: float = DEFAULT_DELAY
) -> List[str]:
    """
    Convenience function to download AI-generated face images.

    Args:
        count: Number of images to download
        output_dir: Directory to save images to
        filename_prefix: Prefix for the image filenames
        delay: Base delay between requests in seconds

    Returns:
        List of successfully downloaded image file paths
    """
    scraper = ImageScraper(img_dir=output_dir, delay=delay)
    return await scraper.download_multiple_images(count, filename_prefix)


def run_face_scraper(
    count: int = 5,
    output_dir: str = "img",
    filename_prefix: str = "ai-face"
) -> List[str]:
    """
    Run the face scraper synchronously.
    
    Args:
        count: Number of images to download
        output_dir: Directory to save images to
        filename_prefix: Prefix for image filenames
    
    Returns:
        List of downloaded image file paths
    """
    config = {
        'url': DEFAULT_URL,
        'rate_limit': DEFAULT_DELAY,
        'requires_js': False
    }
    
    scraper = ImageScraper(config=config, img_dir=output_dir)
    return asyncio.run(scraper.download_multiple_images(count, filename_prefix))


if __name__ == "__main__":
    print("AI-generated face scraper")
    print("Downloading faces...")
    downloaded_images = run_face_scraper()
    print(f"Successfully downloaded {len(downloaded_images)} images.") 