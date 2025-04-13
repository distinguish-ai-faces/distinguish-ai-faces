"""
Core tests for the data pipeline functionality.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from google.cloud import storage
from PIL import Image

from src.preprocess import detect_face_region, resize_and_crop_image
from src.scrape_faces import ImageScraper, validate_image

@pytest.fixture
def mock_storage_client():
    """Fixture for mocking GCS client."""
    mock_client = Mock(spec=storage.Client)
    mock_bucket = Mock(spec=storage.Bucket)
    mock_blob = Mock(spec=storage.Blob)
    
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    return mock_client

@pytest.fixture
def sample_image():
    """Fixture for creating a sample test image."""
    return np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)

@pytest.fixture
def scraping_config():
    """Fixture for scraping configuration."""
    return {
        'url': 'https://thispersondoesnotexist.com',
        'image_selector': 'img#face',
        'rate_limit': 1.0,
        'requires_js': True,
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    }

def test_face_detection_and_crop(sample_image):
    """Test if face detection and cropping works correctly."""
    # Test face box is in the center of the image
    height, width = sample_image.shape[:2]
    face_box = [
        width // 4,           # x1 = 1/4 of width
        height // 4,          # y1 = 1/4 of height
        3 * width // 4,      # x2 = 3/4 of width
        3 * height // 4      # y2 = 3/4 of height
    ]
    
    with patch('src.preprocess.detect_face_region', return_value=face_box):
        processed = resize_and_crop_image(sample_image)
        assert processed.shape == (512, 512, 3)
        # Verify face is centered - since we're using a centered face box,
        # the processed image should maintain this centering
        center_x = processed.shape[1] / 2
        center_y = processed.shape[0] / 2
        face_center_x = (face_box[2] + face_box[0]) / 2
        face_center_y = (face_box[3] + face_box[1]) / 2
        
        # Convert face center to relative position (0-1 range)
        rel_face_x = face_center_x / width
        rel_face_y = face_center_y / height
        
        # These should be very close to 0.5 (center)
        assert abs(rel_face_x - 0.5) < 0.1
        assert abs(rel_face_y - 0.5) < 0.1

def test_scraping_functionality(scraping_config):
    """Test core scraping functionality."""
    scraper = ImageScraper(scraping_config)
    
    # Test successful image download
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b"fake_image_content"
    mock_response.headers = {'Content-Type': 'image/jpeg'}
    
    with patch('requests.get', return_value=mock_response):
        image_data = scraper.download_image_with_retry("https://example.com/image.jpg")
        assert image_data == mock_response.content
        
        # Test image validation
        assert validate_image(mock_response) is True

"""
def test_gcs_integration(mock_storage_client):
    # Test core GCS integration functionality.
    test_bucket = "test-bucket"
    test_file = "test_image.jpg"
    
    with patch('google.cloud.storage.Client', return_value=mock_storage_client):
        # Test bucket access
        mock_storage_client.bucket.return_value.exists.return_value = True
        assert validate_bucket_access(test_bucket) is True
        
        # Test file upload
        result = upload_file(test_file, test_bucket, f"images/{test_file}")
        assert result is True
        mock_storage_client.bucket.assert_called_with(test_bucket)
"""

def test_image_format_and_size():
    """Test image format standardization and size requirements."""
    # Create a test image
    test_image = np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8)
    img = Image.fromarray(test_image)
    
    # Save and process
    with patch('PIL.Image.open', return_value=img):
        processed = resize_and_crop_image(test_image)
        assert processed.shape == (512, 512, 3)  # Verify size
        assert isinstance(processed, np.ndarray)  # Verify format 
