"""
Tests for data integrity and dataset balance.
"""
import pytest
import numpy as np
from pathlib import Path
from ..data_integrity import (
    check_class_balance,
    detect_duplicates,
    validate_dataset_structure,
    compute_image_hash
)

@pytest.fixture
def sample_dataset_structure():
    """Fixture for sample dataset structure."""
    return {
        'real_faces': ['img1.jpg', 'img2.jpg', 'img3.jpg'],
        'ai_faces': ['ai1.jpg', 'ai2.jpg', 'ai3.jpg']
    }

def test_class_balance():
    """Test if dataset classes are balanced."""
    real_count = 100
    ai_count = 100
    
    is_balanced = check_class_balance(real_count, ai_count, threshold=0.1)
    assert is_balanced is True
    
    # Test imbalanced case
    real_count = 100
    ai_count = 70
    is_balanced = check_class_balance(real_count, ai_count, threshold=0.1)
    assert is_balanced is False

def test_duplicate_detection():
    """Test duplicate image detection."""
    # Simulate image hashes
    image_hashes = [
        "hash1",
        "hash1",  # Duplicate
        "hash2",
        "hash3"
    ]
    
    duplicates = detect_duplicates(image_hashes)
    assert len(duplicates) == 1
    assert "hash1" in duplicates

def test_dataset_structure_validation(sample_dataset_structure):
    """Test dataset directory structure validation."""
    is_valid = validate_dataset_structure(sample_dataset_structure)
    assert is_valid is True
    
    # Test invalid structure
    invalid_structure = {
        'real_faces': ['img1.jpg'],
        'unknown_category': ['img2.jpg']
    }
    is_valid = validate_dataset_structure(invalid_structure)
    assert is_valid is False

def test_image_hash_computation():
    """Test image hash computation."""
    # Create a simple test image
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[40:60, 40:60] = 255  # White square in center
    
    hash1 = compute_image_hash(test_image)
    
    # Create slightly modified image
    test_image_modified = test_image.copy()
    test_image_modified[41:59, 41:59] = 255
    
    hash2 = compute_image_hash(test_image_modified)
    
    # Hashes should be similar but not identical
    assert hash1 != hash2
    assert isinstance(hash1, str)
    assert isinstance(hash2, str)

@pytest.mark.parametrize("file_extension", [
    ".jpg",
    ".jpeg",
    ".png",
    ".invalid"
])
def test_file_extension_validation(file_extension):
    """Test validation of file extensions."""
    test_file = Path(f"test_image{file_extension}")
    is_valid = test_file.suffix.lower() in ['.jpg', '.jpeg', '.png']
    
    if file_extension in ['.jpg', '.jpeg', '.png']:
        assert is_valid is True
    else:
        assert is_valid is False

def test_dataset_statistics():
    """Test dataset statistics computation."""
    dataset_stats = {
        'total_images': 200,
        'real_faces': 100,
        'ai_faces': 100,
        'mean_size_mb': 0.5,
        'duplicates': 0
    }
    
    assert dataset_stats['total_images'] == dataset_stats['real_faces'] + dataset_stats['ai_faces']
    assert dataset_stats['duplicates'] == 0
    assert dataset_stats['mean_size_mb'] > 0 