"""
Data processing module for AI face detection.
Handles loading, preprocessing, and augmenting image data.
"""
import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.model_selection import train_test_split
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processor for AI face detection dataset.
    Handles loading, preprocessing, and augmenting image data.
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.1,
        use_gcp: bool = False,
        gcp_bucket_name: Optional[str] = None,
        use_augmentation: bool = True,
        use_enhanced_preprocessing: bool = True  # Gelişmiş ön işleme parametresi eklendi
    ):
        """
        Initialize the data processor.
        
        Args:
            img_size: Target image size (height, width)
            batch_size: Batch size for training
            val_split: Validation split ratio
            test_split: Test split ratio
            use_gcp: Whether to use GCP for data storage
            gcp_bucket_name: GCP bucket name
            use_augmentation: Whether to use data augmentation
            use_enhanced_preprocessing: Whether to use enhanced preprocessing
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.use_gcp = use_gcp
        self.gcp_bucket_name = gcp_bucket_name
        self.use_augmentation = use_augmentation
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        
        # Data caches
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        logger.info(f"Initialized DataProcessor with img_size={img_size}, batch_size={batch_size}")
        logger.info(f"Using GCP: {use_gcp}, Bucket: {gcp_bucket_name if use_gcp else 'N/A'}")
        logger.info(f"Using data augmentation: {use_augmentation}")
        logger.info(f"Using enhanced preprocessing: {use_enhanced_preprocessing}")
    
    def _download_from_gcp(
        self,
        local_dir: Union[str, Path],
        gcp_folder: str
    ) -> Path:
        """
        Download data from GCP bucket to local directory.
        
        Args:
            local_dir: Local directory to download to
            gcp_folder: Folder in GCP bucket
            
        Returns:
            Path to the local download directory
        """
        if not self.use_gcp or not self.gcp_bucket_name:
            logger.error("GCP not configured for download")
            raise ValueError("GCP not configured for download")
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Downloading from GCP bucket {self.gcp_bucket_name}, folder {gcp_folder} to {local_dir}")
        
        try:
            client = storage.Client(project='wingie-devops-project')
            bucket = client.bucket(self.gcp_bucket_name)
            blobs = list(bucket.list_blobs(prefix=gcp_folder))
            
            if not blobs:
                logger.warning(f"No files found in GCP bucket {self.gcp_bucket_name}, folder {gcp_folder}")
                return local_dir
            
            logger.info(f"Found {len(blobs)} files in GCP bucket")
            
            for blob in blobs:
                # Skip directory markers
                if blob.name.endswith('/'):
                    continue
                
                # Extract the relative path from the GCP folder
                rel_path = os.path.relpath(blob.name, gcp_folder)
                local_file_path = local_dir / rel_path
                
                # Ensure the parent directory exists
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download the file
                blob.download_to_filename(str(local_file_path))
                logger.debug(f"Downloaded {blob.name} to {local_file_path}")
            
            logger.info(f"Downloaded {len(blobs)} files from GCP bucket")
            return local_dir
        
        except Exception as e:
            logger.error(f"Error downloading from GCP: {str(e)}")
            raise
    
    def _create_dataset_from_directory(
        self,
        directory: Union[str, Path],
        image_size: Tuple[int, int],
        batch_size: int,
        validation_split: Optional[float] = None,
        subset: Optional[str] = None,
        seed: int = 123
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from a directory of images.
        
        Args:
            directory: Directory containing images
            image_size: Target image size
            batch_size: Batch size
            validation_split: Validation split ratio (if splitting)
            subset: 'training' or 'validation' if using validation_split
            seed: Random seed for reproducibility
            
        Returns:
            tf.data.Dataset object
        """
        return image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='binary',
            color_mode='rgb',
            batch_size=batch_size,
            image_size=image_size,
            shuffle=True,
            seed=seed,
            validation_split=validation_split,
            subset=subset,
            interpolation='bilinear'
        )
    
    def apply_clahe(self, image):
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        This improves contrast in the image, which can help with feature detection.
        
        Args:
            image: Input image tensor
            
        Returns:
            Enhanced image
        """
        # Convert to YUV color space (Y is luminance)
        image_yuv = tf.image.rgb_to_yuv(image)
        
        # Split the channels
        y, u, v = tf.split(image_yuv, 3, axis=-1)
        
        # Normalize y channel to [0, 1]
        y_norm = tf.cast(y, tf.float32) / 255.0
        
        # Apply histogram equalization to y channel (luminance)
        y_eq = tf.image.per_image_standardization(y_norm)
        
        # Scale back to [0, 255]
        y_eq = (y_eq * 255.0)
        
        # Clip values to valid range
        y_eq = tf.clip_by_value(y_eq, 0, 255)
        
        # Önemli değişiklik: uint8'i float32'ye çevir
        # y_eq = tf.cast(y_eq, tf.uint8)  # Eski kod
        y_eq = tf.cast(y_eq, tf.float32)  # Yeni kod - u ve v ile aynı türde
        
        # u ve v'nin de float32 olduğundan emin olalım
        u = tf.cast(u, tf.float32)
        v = tf.cast(v, tf.float32)
        
        # Merge the channels back
        image_eq = tf.concat([y_eq, u, v], axis=-1)
        
        # Convert back to RGB
        image_rgb = tf.image.yuv_to_rgb(image_eq)
        
        return image_rgb
        
    def enhance_sharpness(self, image):
        """
        Enhance image sharpness using unsharp masking.
        
        Args:
            image: Input image tensor
            
        Returns:
            Sharpened image
        """
        # Convert to float32
        image_float = tf.cast(image, tf.float32)
        
        # Daha basit bir bulanıklaştırma yaklaşımı
        # TensorFlow 2.x'in pool fonksiyonları ile yapacağız
        
        # Bir batch boyutu ekleyelim (gerekli)
        img_4d = tf.expand_dims(image_float, 0)
        
        # Average pooling ile bulanıklaştırma - bu bize yaklaşık bir Gaussian blur etkisi verir
        blurred_4d = tf.nn.avg_pool2d(
            img_4d,
            ksize=[1, 5, 5, 1],  # 5x5 kernel
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        
        # Batch boyutunu geri kaldıralım
        blurred = tf.squeeze(blurred_4d, 0)
        
        # Detay katmanını hesaplayalım (orijinal - bulanık)
        detail = image_float - blurred
        
        # Detay katmanını orijinale ekleyelim (keskinleştirme)
        amount = 1.5  # Keskinleştirme şiddeti
        sharpened = image_float + amount * detail
        
        # Değerleri geçerli aralığa kırpalım
        sharpened = tf.clip_by_value(sharpened, 0, 255)
        
        # uint8'e geri dönüştürelim
        sharpened = tf.cast(sharpened, tf.uint8)
        
        return sharpened
    
    def preprocess_image(self, image, label):
        """
        Apply custom preprocessing to an image.
        
        Args:
            image: Input image tensor
            label: Image label
            
        Returns:
            Tuple of (preprocessed image, label)
        """
        if not self.use_enhanced_preprocessing:
            return image, label
            
        try:
            # Get image shape and ensure it's 3D [height, width, channels]
            # If it's already 4D [batch, height, width, channels], we need to handle differently
            shape = tf.shape(image)
            rank = tf.rank(image)
            
            # If rank is 4 (already has batch dimension), we need to squeeze it first
            if rank == 4:
                image = tf.squeeze(image, axis=0)
            
            # Convert to float32 for processing
            image = tf.cast(image, tf.float32)
            
            # Basitleştirilmiş ön işleme - sadece normalizasyon yapalım
            # CLAHE ve keskinleştirme atlanıyor çünkü çok fazla tensor şekil sorunu yaratıyorlar
            
            # Normalize pixel values to [0, 1]
            image = image / 255.0
            
            # Scale back to [0, 255] and convert to uint8
            image = tf.clip_by_value(image * 255.0, 0, 255)
            image = tf.cast(image, tf.uint8)
            
            return image, label
            
        except Exception as e:
            # Bir hatayla karşılaşırsak, gelişmiş ön işlemeyi atla ve orijinal görüntüyü döndür
            tf.print("Ön işleme hatası:", e)
            return image, label

    def load_dataset(
        self,
        data_dir: Union[str, Path] = None,
        gcp_ai_folder: str = 'ai-faces',
        gcp_human_folder: str = 'human-faces',
        local_ai_dir: Union[str, Path] = 'data/ai_faces',
        local_human_dir: Union[str, Path] = 'data/human_faces'
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load and prepare the dataset.
        
        Args:
            data_dir: Base data directory (local)
            gcp_ai_folder: GCP folder for AI faces
            gcp_human_folder: GCP folder for human faces
            local_ai_dir: Local directory for AI faces
            local_human_dir: Local directory for human faces
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Set up directories
        if data_dir:
            base_dir = Path(data_dir)
        else:
            base_dir = Path.cwd() / 'data'
        
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert local dirs to paths
        local_ai_dir = base_dir / local_ai_dir
        local_human_dir = base_dir / local_human_dir
        
        # Download from GCP if needed
        if self.use_gcp:
            logger.info("Downloading data from GCP bucket")
            local_ai_dir = self._download_from_gcp(local_ai_dir, gcp_ai_folder)
            local_human_dir = self._download_from_gcp(local_human_dir, gcp_human_folder)
        
        # Check if directories exist and contain files
        if not local_ai_dir.exists() or not list(local_ai_dir.glob('*.jpg')):
            raise ValueError(f"AI faces directory {local_ai_dir} doesn't exist or is empty")
        
        if not local_human_dir.exists() or not list(local_human_dir.glob('*.jpg')):
            raise ValueError(f"Human faces directory {local_human_dir} doesn't exist or is empty")
        
        # Prepare merged dataset with proper labels
        ai_output_dir = base_dir / 'processed' / 'ai'
        human_output_dir = base_dir / 'processed' / 'human'
        
        # Create label directories if they don't exist
        ai_output_dir.mkdir(parents=True, exist_ok=True)
        human_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks for AI images (labeled 0)
        for img_path in local_ai_dir.glob('*.jpg'):
            target_path = ai_output_dir / img_path.name
            if not target_path.exists():
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(img_path, target_path)
                else:  # Unix-like
                    target_path.symlink_to(img_path)
        
        # Create symlinks for human images (labeled 1)
        for img_path in local_human_dir.glob('*.jpg'):
            target_path = human_output_dir / img_path.name
            if not target_path.exists():
                if os.name == 'nt':  # Windows
                    import shutil
                    shutil.copy2(img_path, target_path)
                else:  # Unix-like
                    target_path.symlink_to(img_path)
        
        # Count images
        ai_count = len(list(ai_output_dir.glob('*.jpg')))
        human_count = len(list(human_output_dir.glob('*.jpg')))
        logger.info(f"Prepared dataset with {ai_count} AI faces and {human_count} human faces")
        
        # ----- Başlangıçta daha güvenli bir yaklaşım kullanılacak -----
        # TensorFlow'un yerleşik image_dataset_from_directory yerine
        # kendi veri yükleme ve işleme yaklaşımımızı deneyelim
        
        # Tüm görüntüleri ve etiketleri yükle
        all_images = []
        all_labels = []
        
        # AI görüntülerini yükle (etiket 0)
        for img_path in sorted(ai_output_dir.glob('*.jpg')):
            try:
                # Görüntüyü oku ve yeniden boyutlandır
                img = tf.io.read_file(str(img_path))
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, self.img_size, method='bilinear')
                img = tf.cast(img, tf.uint8)  # uint8 olarak sakla
                
                all_images.append(img)
                all_labels.append(0)  # AI etiketi
            except Exception as e:
                logger.warning(f"Görüntü yüklenirken hata: {img_path}, {str(e)}")
        
        # Human görüntülerini yükle (etiket 1)
        for img_path in sorted(human_output_dir.glob('*.jpg')):
            try:
                # Görüntüyü oku ve yeniden boyutlandır
                img = tf.io.read_file(str(img_path))
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, self.img_size, method='bilinear')
                img = tf.cast(img, tf.uint8)  # uint8 olarak sakla
                
                all_images.append(img)
                all_labels.append(1)  # Human etiketi
            except Exception as e:
                logger.warning(f"Görüntü yüklenirken hata: {img_path}, {str(e)}")
        
        # Liste boş kontrolü
        if not all_images or not all_labels:
            raise ValueError("Görüntü yüklenemedi veya görüntü listesi boş")
        
        # NumPy dizisine dönüştür
        all_images = np.array(all_images, dtype=np.uint8)
        all_labels = np.array(all_labels, dtype=np.float32)
        
        # Veriyi karıştır
        indices = np.random.permutation(len(all_images))
        all_images = all_images[indices]
        all_labels = all_labels[indices]
        
        # Eğitim/doğrulama/test bölünmesi
        val_test_size = self.val_split + self.test_split
        train_size = int(len(all_images) * (1 - val_test_size))
        val_size = int(len(all_images) * self.val_split)
        
        # Bölünme
        train_images = all_images[:train_size]
        train_labels = all_labels[:train_size]
        
        val_images = all_images[train_size:train_size+val_size]
        val_labels = all_labels[train_size:train_size+val_size]
        
        test_images = all_images[train_size+val_size:]
        test_labels = all_labels[train_size+val_size:]
        
        # TensorFlow veri setlerine dönüştür
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
        val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
        
        # Batch haline getir
        train_ds = train_ds.batch(self.batch_size)
        val_ds = val_ds.batch(self.batch_size)
        test_ds = test_ds.batch(self.batch_size)
        
        # Veri artırma (eğitim verisi için)
        if self.use_augmentation:
            logger.info("Applying data augmentation to training dataset")
            train_ds = self.apply_augmentation(train_ds)
        
        # Optimize et
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
        
        # Veri setlerini önbelleğe al
        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds
        
        logger.info(f"Dataset prepared: {len(train_ds)} training, {len(val_ds)} validation, {len(test_ds)} test batches")
        
        return train_ds, val_ds, test_ds
    
    def get_data_augmentation(self) -> ImageDataGenerator:
        """
        Create an advanced image data augmentation generator.
        
        Returns:
            ImageDataGenerator for data augmentation
        """
        return ImageDataGenerator(
            # More aggressive rotation
            rotation_range=40,  # Increased from 20
            
            # More aggressive shifts
            width_shift_range=0.3,  # Increased from 0.2
            height_shift_range=0.3,  # Increased from 0.2
            
            # More aggressive shear
            shear_range=0.3,  # Increased from 0.2
            
            # More aggressive zoom
            zoom_range=0.3,  # Increased from 0.2
            
            # Flips
            horizontal_flip=True,
            vertical_flip=False,  # Not typically useful for face images
            
            # Color augmentations
            brightness_range=[0.7, 1.3],  # 70-130% brightness
            channel_shift_range=0.2,  # RGB color channel shifts
            
            # Fill mode for transformations
            fill_mode='nearest',
            
            # Add some random noise
            preprocessing_function=self._add_random_noise
        )
    
    def _add_random_noise(self, img):
        """
        Add random noise to an image for data augmentation.
        
        Args:
            img: Input image tensor
            
        Returns:
            Augmented image with noise
        """
        # Add random Gaussian noise
        if np.random.rand() > 0.5:  # 50% chance to add noise
            intensity = np.random.uniform(0.01, 0.05)  # Random noise intensity
            noise = np.random.normal(0, intensity, img.shape)
            img = img + noise
            img = np.clip(img, 0, 1.0)  # Ensure values stay in [0,1]
        
        # Random contrast adjustment
        if np.random.rand() > 0.5:  # 50% chance to adjust contrast
            contrast_factor = np.random.uniform(0.7, 1.3)
            img = (img - 0.5) * contrast_factor + 0.5
            img = np.clip(img, 0, 1.0)
        
        return img

    def apply_augmentation(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """
        Apply data augmentation to a TensorFlow dataset.
        Uses a simplified approach with basic, reliable TensorFlow operations.
        
        Args:
            dataset: The input dataset
            
        Returns:
            Augmented dataset
        """
        logger.info("Creating augmentation function")
        
        def simple_augment(image, label):
            """Extremely simplified augmentation function to avoid shape issues"""
            # Ensure static shape
            image = tf.ensure_shape(image, [*self.img_size, 3])
            
            # Convert to float for processing
            image = tf.cast(image, tf.float32)
            
            # Random flip - 50% chance
            if tf.random.uniform(()) > 0.5:
                image = tf.image.flip_left_right(image)
            
            # Apply random brightness and contrast adjustments
            image = tf.image.random_brightness(image, 0.1)
            image = tf.image.random_contrast(image, 0.8, 1.2)
            
            # Ensure range is correct
            image = tf.clip_by_value(image, 0.0, 255.0)
            
            # Convert back to uint8
            image = tf.cast(image, tf.uint8)
            
            return image, label
        
        try:
            # Map the augmentation function
            augmented_dataset = dataset.map(
                simple_augment,
                num_parallel_calls=tf.data.AUTOTUNE
            )
            
            logger.info("Simple data augmentation applied successfully")
            return augmented_dataset
        
        except Exception as e:
            # If augmentation fails, return the original dataset
            logger.error(f"Error applying augmentation: {e}, returning original dataset")
            return dataset 