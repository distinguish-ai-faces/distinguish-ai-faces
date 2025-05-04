"""
Image preprocessing module.
Handles operations for standardizing and preparing images for model training.
"""
import os
import logging
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default output dimensions
DEFAULT_OUTPUT_SIZE = (512, 512)
DEFAULT_PADDING_COLOR = (0, 0, 0)  # Black padding


def crop_watermark(
    image_data: Union[bytes, np.ndarray, Image.Image],
    watermark_height: int = 24
) -> Union[bytes, np.ndarray, Image.Image]:
    """
    Crop the watermark from an image.
    
    Args:
        image_data: Image data (bytes, numpy array, or PIL.Image)
        watermark_height: Height of the watermark to remove from bottom

    Returns:
        Cropped image in the same format as input
    """
    # Handle different input types
    return_bytes = False
    if isinstance(image_data, bytes):
        return_bytes = True
        img = Image.open(BytesIO(image_data))
    elif isinstance(image_data, np.ndarray):
        img = Image.fromarray(image_data)
    elif isinstance(image_data, Image.Image):
        img = image_data
    else:
        raise TypeError(f"Unsupported image data type: {type(image_data)}")
    
    # Crop the watermark
    width, height = img.size
    if height <= watermark_height:
        logger.warning("Image height is less than or equal to watermark height, skipping crop")
        return image_data
    
    cropped_img = img.crop((0, 0, width, height - watermark_height))
    
    # Return in the same format as input
    if return_bytes:
        output = BytesIO()
        cropped_img.save(output, format='JPEG', quality=95)
        return output.getvalue()
    elif isinstance(image_data, np.ndarray):
        return np.array(cropped_img)
    else:
        return cropped_img


def detect_face_region(
    image: np.ndarray,
    face_cascade_path: str = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
) -> Optional[List[int]]:
    """
    Detect the face region in an image.
    
    Args:
        image: Input image as numpy array
        face_cascade_path: Path to the face cascade XML file

    Returns:
        List [x1, y1, x2, y2] with face coordinates or None if no face detected
    """
    # Ensure image is in grayscale for face detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Load the face cascade
    try:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return None
    
    # Return the first face detected
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return [x, y, x + w, y + h]
    
    return None


def validate_dimensions(
    image: np.ndarray,
    min_dims: Tuple[int, int] = (64, 64),
    max_dims: Tuple[int, int] = (4096, 4096)
) -> bool:
    """
    Validate if image dimensions are within acceptable range.
    
    Args:
        image: Input image as numpy array
        min_dims: Minimum dimensions (width, height)
        max_dims: Maximum dimensions (width, height)

    Returns:
        Boolean indicating if dimensions are valid
    """
    height, width = image.shape[:2]
    min_width, min_height = min_dims
    max_width, max_height = max_dims
    
    return (
        width >= min_width and 
        height >= min_height and
        width <= max_width and
        height <= max_height
    )


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image pixel values to [0, 1] range.
    
    Args:
        image: Input image as numpy array

    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def resize_and_crop_image(
    image: Union[np.ndarray, str, Path],
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    center_face: bool = True,
    padding_color: Tuple[int, int, int] = DEFAULT_PADDING_COLOR
) -> np.ndarray:
    """
    Resize and crop image to the desired dimensions.
    
    Args:
        image: Input image (numpy array or path to image file)
        output_size: Desired output size (width, height)
        center_face: Whether to center the face in the crop
        padding_color: Color for padding (R, G, B)

    Returns:
        Processed image as numpy array
    """
    # Load image if it's a path
    if isinstance(image, (str, Path)):
        try:
            image = cv2.imread(str(image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            raise
    
    # Validate image
    if not validate_dimensions(image):
        logger.warning(f"Image dimensions outside valid range: {image.shape[:2]}")
    
    # Get output dimensions
    output_width, output_height = output_size
    
    # Determine crop region
    face_region = None
    if center_face:
        face_region = detect_face_region(image)
    
    if face_region:
        # Center crop around face
        face_x1, face_y1, face_x2, face_y2 = face_region
        face_center_x = (face_x1 + face_x2) // 2
        face_center_y = (face_y1 + face_y2) // 2
        
        # Face size with some margin
        face_width = int((face_x2 - face_x1) * 1.5)
        face_height = int((face_y2 - face_y1) * 1.5)
        
        # Make it square by taking the larger dimension
        crop_size = max(face_width, face_height)
        
        # Calculate crop coordinates
        crop_x1 = max(0, face_center_x - crop_size // 2)
        crop_y1 = max(0, face_center_y - crop_size // 2)
        crop_x2 = min(image.shape[1], crop_x1 + crop_size)
        crop_y2 = min(image.shape[0], crop_y1 + crop_size)
        
        # Adjust if crop is out of bounds
        if crop_x2 - crop_x1 < crop_size:
            crop_x1 = max(0, crop_x2 - crop_size)
        if crop_y2 - crop_y1 < crop_size:
            crop_y1 = max(0, crop_y2 - crop_size)
        
        # Crop the image
        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        # Default to center crop if no face detected
        height, width = image.shape[:2]
        
        # Make it square by taking the smaller dimension
        crop_size = min(width, height)
        
        # Center crop
        start_x = (width - crop_size) // 2
        start_y = (height - crop_size) // 2
        
        cropped = image[start_y:start_y+crop_size, start_x:start_x+crop_size]
    
    # Resize to the desired dimensions
    resized = cv2.resize(cropped, (output_width, output_height), interpolation=cv2.INTER_AREA)
    
    return resized


def batch_process_images(
    images: List[Union[np.ndarray, str, Path]],
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    center_face: bool = True,
    max_workers: int = 4
) -> List[np.ndarray]:
    """
    Process a batch of images.
    
    Args:
        images: List of input images or paths
        output_size: Desired output size for all images
        center_face: Whether to center faces in crops
        max_workers: Maximum number of worker threads for parallel processing

    Returns:
        List of processed images
    """
    if len(images) == 1:
        # Process a single image without parallel processing
        return [resize_and_crop_image(images[0], output_size, center_face)]
    
    # Process multiple images in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use a lambda to pass additional arguments
        process_func = lambda img: resize_and_crop_image(img, output_size, center_face)
        results = list(executor.map(process_func, images))
    
    return results


def is_exact_duplicate(img1: np.ndarray, img2: np.ndarray) -> bool:
    """
    Check if two images are exact duplicates by comparing pixels directly.
    
    Args:
        img1: First image as numpy array
        img2: Second image as numpy array
        
    Returns:
        Boolean indicating if images are exact duplicates
    """
    if img1.shape != img2.shape:
        return False
    
    # Compare arrays directly
    return np.array_equal(img1, img2)


def process_scraped_images(
    input_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    output_size: Tuple[int, int] = DEFAULT_OUTPUT_SIZE,
    center_face: bool = True,
    remove_watermark: bool = True,
    target_format: str = 'jpg',
    quality: int = 95
) -> List[Dict[str, Any]]:
    """
    Process all images in the scraped images directory.
    
    Args:
        input_dir: Directory containing scraped images
        output_dir: Directory to save processed images (if None, use input_dir)
        output_size: Desired output size
        center_face: Whether to center faces in crops
        remove_watermark: Whether to remove watermarks
        target_format: Output format for the processed images
        quality: JPEG quality (0-100)

    Returns:
        List of dictionaries with processing results
    """
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir / "processed"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in input directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_files = [
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    results = []
    
    for img_path in image_files:
        try:
            # Dosya adını koru - sadece uzantıyı target_format olarak değiştir
            output_path = output_dir / f"{img_path.stem}.{target_format}"
            
            # Eğer output_path zaten varsa, işlemeyi atla
            if output_path.exists():
                logger.info(f"Skipping {img_path} as {output_path} already exists")
                results.append({
                    'input_path': str(img_path),
                    'output_path': str(output_path),
                    'success': True,
                    'skipped': True
                })
                continue
            
            # Load image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            
            # Process image
            if remove_watermark:
                img_pil = Image.fromarray(img)
                img_pil = crop_watermark(img_pil)
                img = np.array(img_pil)
            
            processed_img = resize_and_crop_image(img, output_size, center_face)
            
            # Save processed image
            processed_pil = Image.fromarray(processed_img)
            if target_format.lower() in ('jpg', 'jpeg'):
                processed_pil.save(output_path, format='JPEG', quality=quality)
            else:
                processed_pil.save(output_path, format=target_format.upper())
            
            results.append({
                'input_path': str(img_path),
                'output_path': str(output_path),
                'success': True
            })
            
            logger.info(f"Processed {img_path} -> {output_path}")
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
            results.append({
                'input_path': str(img_path),
                'success': False,
                'error': str(e)
            })
    
    # Log summary
    success_count = sum(1 for r in results if r['success'])
    skipped_count = sum(1 for r in results if r.get('success') and r.get('skipped'))
    logger.info(f"Processed {success_count}/{len(results)} images successfully (skipped: {skipped_count})")
    
    return results


if __name__ == "__main__":
    print("Image preprocessing utility")
    print("Use this module's functions to preprocess images") 