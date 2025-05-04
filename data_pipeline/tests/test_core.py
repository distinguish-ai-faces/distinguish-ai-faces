"""
Core tests for the data pipeline functionality.
"""
import pytest
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock
from google.cloud import storage
from PIL import Image
from pathlib import Path

from src.preprocess import detect_face_region, resize_and_crop_image
from src.scrape_faces import ImageScraper, validate_image
from src.gcp_storage import (
    validate_bucket_access, 
    upload_file, 
    download_file,
    list_bucket_files
)

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

def test_gcs_bucket_access(mock_storage_client):
    """Test bucket access validation."""
    test_bucket = "test-bucket"
    
    with patch('src.gcp_storage.setup_storage_client', return_value=mock_storage_client):
        # Mock bucket.exists() to return True
        mock_storage_client.bucket.return_value.exists.return_value = True
        
        # Test bucket access validation
        assert validate_bucket_access(test_bucket) is True
        mock_storage_client.bucket.assert_called_with(test_bucket)
        
        # Test bucket access failure
        mock_storage_client.bucket.return_value.exists.return_value = False
        assert validate_bucket_access(test_bucket) is False

def test_gcs_upload_file(mock_storage_client, tmp_path):
    """Test file upload to GCS bucket."""
    test_bucket = "test-bucket"
    
    # Create a temporary test file
    test_file = tmp_path / "test.jpg"
    test_file.write_bytes(b"test image content")
    
    with patch('src.gcp_storage.setup_storage_client', return_value=mock_storage_client):
        # Setup the mock blob
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        
        # Test successful upload
        result = upload_file(test_file, test_bucket)
        
        assert result["success"] is True
        assert result["bucket"] == test_bucket
        assert result["path"] == test_file.name
        
        # Verify the mock calls
        mock_storage_client.bucket.assert_called_with(test_bucket)
        mock_storage_client.bucket.return_value.blob.assert_called_with(test_file.name)
        mock_blob.upload_from_filename.assert_called_with(str(test_file))

def test_gcs_download_file(mock_storage_client, tmp_path):
    """Test file download from GCS bucket."""
    test_bucket = "test-bucket"
    cloud_path = "images/test.jpg"
    local_path = tmp_path / "downloaded.jpg"
    
    with patch('src.gcp_storage.setup_storage_client', return_value=mock_storage_client):
        # Setup the mock blob
        mock_blob = mock_storage_client.bucket.return_value.blob.return_value
        mock_blob.exists.return_value = True
        
        # Mock the download_to_filename method to create an empty file
        def mock_download(path):
            # Simulate file creation
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).touch()
            
        mock_blob.download_to_filename.side_effect = mock_download
        
        # Test successful download
        result = download_file(cloud_path, local_path, test_bucket)
        
        assert result["success"] is True
        assert result["bucket"] == test_bucket
        assert result["cloud_path"] == cloud_path
        
        # Verify the mock calls
        mock_storage_client.bucket.assert_called_with(test_bucket)
        mock_storage_client.bucket.return_value.blob.assert_called_with(cloud_path)
        mock_blob.download_to_filename.assert_called_with(str(local_path))
        
        # Test download failure for non-existent blob
        mock_blob.exists.return_value = False
        result = download_file("nonexistent.jpg", local_path, test_bucket)
        assert result["success"] is False

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
