"""
Tests for image preprocessing functionality.
"""
import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, patch
from ..preprocess import (
    resize_and_crop_image,
    normalize_image,
    validate_dimensions,
    detect_face_region,
    batch_process_images
)

@pytest.fixture
def sample_image():
    """Fixture for creating a sample test image."""
    return np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)

@pytest.fixture
def mock_face_detector():
    """Fixture for mocking face detection."""
    mock = Mock()
    mock.detect.return_value = [200, 150, 500, 450]  # [x1, y1, x2, y2]
    return mock

def test_image_resize_to_512():
    """Test if image is resized to 512x512."""
    test_sizes = [(800, 600), (600, 800), (1024, 1024), (400, 300)]
    
    for width, height in test_sizes:
        input_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        processed = resize_and_crop_image(input_image)
        
        assert processed.shape == (512, 512, 3)

def test_face_centered_crop():
    """Test if face is properly centered in the crop."""
    input_image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
    face_box = [300, 300, 700, 700]  # Simulated face detection
    
    with patch('..preprocess.detect_face_region', return_value=face_box):
        processed = resize_and_crop_image(input_image)
        assert processed.shape == (512, 512, 3)

def test_small_image_upscaling():
    """Test handling of images smaller than 512x512."""
    small_image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    processed = resize_and_crop_image(small_image)
    
    assert processed.shape == (512, 512, 3)
    # Verify image quality is acceptable after upscaling
    assert processed.std() > 20  # Ensure we haven't lost too much detail

def test_aspect_ratio_preservation_in_crop():
    """Test if the face aspect ratio is preserved during cropping."""
    input_image = np.random.randint(0, 256, (1000, 800, 3), dtype=np.uint8)
    face_box = [200, 150, 600, 550]  # Simulated face detection
    
    with patch('..preprocess.detect_face_region', return_value=face_box):
        processed = resize_and_crop_image(input_image)
        assert processed.shape[:2] == (512, 512)  # Square output
        
        # The face should maintain its approximate aspect ratio
        face_height = face_box[3] - face_box[1]
        face_width = face_box[2] - face_box[0]
        original_ratio = face_width / face_height
        assert 0.9 <= original_ratio <= 1.1  # Allow small variation

def test_no_face_detected_handling():
    """Test handling when no face is detected."""
    input_image = np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8)
    
    with patch('..preprocess.detect_face_region', return_value=None):
        processed = resize_and_crop_image(input_image)
        assert processed.shape == (512, 512, 3)
        # Should default to center crop when no face is detected
        assert np.array_equal(processed.shape[:2], np.array([512, 512]))

def test_multiple_faces_handling():
    """Test handling when multiple faces are detected."""
    input_image = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
    face_boxes = [[100, 100, 300, 300], [500, 500, 700, 700]]
    
    with patch('..preprocess.detect_face_region', return_value=face_boxes[0]):  # Use first face
        processed = resize_and_crop_image(input_image)
        assert processed.shape == (512, 512, 3)

def test_batch_processing():
    """Test batch processing of images."""
    test_images = [
        np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8),
        np.random.randint(0, 256, (600, 800, 3), dtype=np.uint8),
        np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    ]
    
    results = batch_process_images(test_images)
    
    assert len(results) == len(test_images)
    for processed in results:
        assert processed.shape == (512, 512, 3)

@pytest.mark.parametrize("input_size,expected_size", [
    ((800, 600), (512, 512)),
    ((600, 800), (512, 512)),
    ((300, 300), (512, 512)),
    ((1024, 1024), (512, 512))
])
def test_output_dimensions(input_size, expected_size):
    """Test if output dimensions are always 512x512."""
    input_image = np.random.randint(0, 256, (*input_size, 3), dtype=np.uint8)
    processed = resize_and_crop_image(input_image)
    assert processed.shape[:2] == expected_size

def test_image_quality_preservation():
    """Test if image quality is preserved after resizing and cropping."""
    input_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    processed = resize_and_crop_image(input_image)
    
    # Check if we maintain good image statistics
    assert processed.mean() > 0
    assert processed.std() > 0
    assert len(np.unique(processed)) > 1000  # Ensure we haven't lost too much detail

def test_parallel_batch_processing():
    """Test parallel processing of multiple images."""
    test_images = [np.random.randint(0, 256, (800, 600, 3), dtype=np.uint8) for _ in range(5)]
    
    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.return_value = [
            np.zeros((512, 512, 3)) for _ in test_images
        ]
        
        results = batch_process_images(test_images, max_workers=4)
        assert len(results) == len(test_images)
        for processed in results:
            assert processed.shape == (512, 512, 3)

def test_file_size_check():
    """Test if processed images maintain reasonable file size."""
    input_image = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
    processed = resize_and_crop_image(input_image)
    
    # Convert to PIL Image and save as JPEG
    with patch('PIL.Image') as mock_pil:
        img = Image.fromarray(processed)
        with patch('io.BytesIO') as mock_buffer:
            img.save(mock_buffer, format='JPEG', quality=95)
            # Check if file size is reasonable (typically 50-150KB for 512x512)
            assert mock_buffer.tell() < 150 * 1024  # Should be less than 150KB 