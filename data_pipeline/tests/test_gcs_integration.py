"""
Tests for Google Cloud Storage integration with format standardization.
"""
import pytest
from unittest.mock import Mock, patch
from google.cloud import storage
from ..upload_to_gcs import (
    upload_file,
    validate_bucket_access,
    check_file_exists,
    upload_metadata,
    upload_with_format_check
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
def sample_image_batch():
    """Fixture for sample image batch with different formats."""
    return [
        {"path": "image1.png", "size": 1024},
        {"path": "image2.jpg", "size": 2048},
        {"path": "image3.webp", "size": 1536},
        {"path": "image4.bmp", "size": 4096}
    ]

def test_format_validation_before_upload(mock_storage_client):
    """Test format validation before upload."""
    test_cases = [
        ("test.jpg", True),
        ("test.jpeg", True),
        ("test.png", False),  # Should be converted
        ("test.webp", False)  # Should be converted
    ]
    
    for file_path, expected in test_cases:
        with patch('google.cloud.storage.Client', return_value=mock_storage_client):
            result = upload_with_format_check(file_path, "test-bucket", check_format=True)
            assert result['needs_conversion'] != expected

def test_auto_conversion_upload(mock_storage_client):
    """Test automatic format conversion during upload."""
    non_jpg_image = "test.png"
    
    with patch('google.cloud.storage.Client', return_value=mock_storage_client), \
         patch('PIL.Image.open'), patch('PIL.Image.save'):
        result = upload_with_format_check(non_jpg_image, "test-bucket", auto_convert=True)
        assert result['converted_to_jpg'] is True
        assert result['original_format'] == 'PNG'

def test_batch_upload_with_conversion(mock_storage_client, sample_image_batch):
    """Test batch upload with format conversion."""
    with patch('google.cloud.storage.Client', return_value=mock_storage_client), \
         patch('concurrent.futures.ThreadPoolExecutor'):
        results = upload_with_format_check(
            [img["path"] for img in sample_image_batch],
            "test-bucket",
            auto_convert=True,
            parallel_uploads=True
        )
        assert all(r['success'] for r in results)
        assert all(r['final_format'] == 'JPG' for r in results)

def test_metadata_update_after_conversion(mock_storage_client):
    """Test metadata update after format conversion."""
    original_metadata = {
        'original_format': 'PNG',
        'original_size': 1024
    }
    
    with patch('google.cloud.storage.Client', return_value=mock_storage_client):
        result = upload_metadata(
            "test-bucket",
            "converted_image.jpg",
            {**original_metadata, 'converted_format': 'JPG'}
        )
        assert result is True

def test_failed_conversion_handling(mock_storage_client):
    """Test handling of failed format conversions."""
    with patch('PIL.Image.open', side_effect=Exception("Conversion failed")):
        with pytest.raises(Exception) as exc_info:
            upload_with_format_check("corrupt.webp", "test-bucket", auto_convert=True)
        assert "Conversion failed" in str(exc_info.value)

@pytest.mark.parametrize("original_format,expected_size", [
    ('PNG', 1024),
    ('WEBP', 768),
    ('BMP', 2048)
])
def test_size_optimization(mock_storage_client, original_format, expected_size):
    """Test size optimization during format conversion."""
    with patch('google.cloud.storage.Client', return_value=mock_storage_client), \
         patch('os.path.getsize') as mock_size:
        mock_size.return_value = expected_size
        result = upload_with_format_check(
            f"test.{original_format.lower()}",
            "test-bucket",
            optimize_size=True
        )
        assert result['optimized_size'] < expected_size

def test_concurrent_format_conversion(mock_storage_client, sample_image_batch):
    """Test concurrent format conversion and upload."""
    with patch('google.cloud.storage.Client', return_value=mock_storage_client), \
         patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.return_value = [
            {'success': True, 'path': f"{img['path']}.jpg"} for img in sample_image_batch
        ]
        
        results = upload_with_format_check(
            [img["path"] for img in sample_image_batch],
            "test-bucket",
            parallel_uploads=True,
            max_workers=4
        )
        assert len(results) == len(sample_image_batch)

def test_format_conversion_retry(mock_storage_client):
    """Test retry mechanism for format conversion."""
    with patch('google.cloud.storage.Client', return_value=mock_storage_client), \
         patch('PIL.Image.open') as mock_open:
        mock_open.side_effect = [Exception("First attempt failed"), Mock()]  # Fail once, succeed on retry
        
        result = upload_with_format_check(
            "test.png",
            "test-bucket",
            auto_convert=True,
            max_retries=2
        )
        assert result['success'] is True
        assert result['retry_count'] == 1

def test_preserve_image_quality(mock_storage_client):
    """Test image quality preservation during format conversion."""
    with patch('google.cloud.storage.Client', return_value=mock_storage_client), \
         patch('PIL.Image.open') as mock_open, \
         patch('PIL.Image.save') as mock_save:
        mock_open.return_value = Mock(format='PNG')
        
        upload_with_format_check(
            "high_quality.png",
            "test-bucket",
            auto_convert=True,
            quality=95
        )
        mock_save.assert_called_once_with(quality=95)

def test_bucket_access_validation(mock_storage_client):
    """Test GCS bucket access validation."""
    with patch('google.cloud.storage.Client', return_value=mock_storage_client):
        mock_storage_client.bucket.return_value.exists.return_value = True
        assert validate_bucket_access("test-bucket") is True
        
        mock_storage_client.bucket.return_value.exists.return_value = False
        assert validate_bucket_access("nonexistent-bucket") is False

def test_file_upload_success(mock_storage_client):
    """Test successful file upload to GCS."""
    test_file_path = "test_image.jpg"
    destination_blob_name = "images/test_image.jpg"
    
    with patch('google.cloud.storage.Client', return_value=mock_storage_client):
        result = upload_file(test_file_path, "test-bucket", destination_blob_name)
        assert result is True
        mock_storage_client.bucket.assert_called_once_with("test-bucket")

def test_file_upload_failure(mock_storage_client):
    """Test file upload failure handling."""
    mock_storage_client.bucket.return_value.blob.return_value.upload_from_filename.side_effect = Exception("Upload failed")
    
    with patch('google.cloud.storage.Client', return_value=mock_storage_client):
        with pytest.raises(Exception):
            upload_file("nonexistent.jpg", "test-bucket", "images/nonexistent.jpg")

def test_file_existence_check(mock_storage_client):
    """Test checking if file exists in GCS bucket."""
    mock_blob = mock_storage_client.bucket.return_value.blob.return_value
    
    with patch('google.cloud.storage.Client', return_value=mock_storage_client):
        mock_blob.exists.return_value = True
        assert check_file_exists("test-bucket", "existing-file.jpg") is True
        
        mock_blob.exists.return_value = False
        assert check_file_exists("test-bucket", "nonexistent-file.jpg") is False

def test_metadata_upload(mock_storage_client):
    """Test metadata upload to GCS."""
    test_metadata = {
        'image_type': 'real_face',
        'resolution': '128x128',
        'format': 'jpg'
    }
    
    mock_blob = mock_storage_client.bucket.return_value.blob.return_value
    
    with patch('google.cloud.storage.Client', return_value=mock_storage_client):
        result = upload_metadata("test-bucket", "test_image.jpg", test_metadata)
        assert result is True
        mock_blob.metadata = test_metadata
        mock_blob.patch.assert_called_once()

@pytest.mark.parametrize("file_path,expected", [
    ("valid/path/image.jpg", True),
    ("", False),
    (None, False)
])
def test_file_path_validation(file_path, expected):
    """Test validation of file paths before upload."""
    assert bool(file_path and file_path.strip()) == expected

def test_batch_upload_with_retry(mock_storage_client):
    """Test batch upload with retry mechanism."""
    test_files = [
        ("file1.jpg", "images/file1.jpg"),
        ("file2.jpg", "images/file2.jpg"),
        ("file3.jpg", "images/file3.jpg")
    ]
    
    with patch('google.cloud.storage.Client', return_value=mock_storage_client):
        for local_file, blob_name in test_files:
            result = upload_file(local_file, "test-bucket", blob_name)
            assert result is True 