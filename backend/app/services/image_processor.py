import os
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import uuid
import logging
from typing import List, Dict, Tuple, Union, Optional
from io import BytesIO
import base64
from pathlib import Path

logger = logging.getLogger(__name__)

class FashionImageProcessor:
    """
    Image processing service for fashion items including cropping, resizing, and preprocessing
    """
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize image processor
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir or "/tmp/fashion_processor"
        self._ensure_temp_dir()
        
    def _ensure_temp_dir(self):
        """Create temp directory if it doesn't exist"""
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def crop_detected_objects(self, image_path: str, detections: List[Dict], 
                            padding: int = 10, min_size: int = 50) -> List[Dict]:
        """
        Crop detected objects from original image
        
        Args:
            image_path: Path to original image
            detections: List of YOLO detection results
            padding: Extra padding around bounding box
            min_size: Minimum size for cropped objects
            
        Returns:
            List of detection results with cropped images and file paths
        """
        try:
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            h, w = image.shape[:2]
            results = []
            
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                
                # Add padding to bounding box
                x1 = max(0, bbox['x1'] - padding)
                y1 = max(0, bbox['y1'] - padding)
                x2 = min(w, bbox['x2'] + padding)
                y2 = min(h, bbox['y2'] + padding)
                
                # Check minimum size
                crop_width = x2 - x1
                crop_height = y2 - y1
                
                if crop_width < min_size or crop_height < min_size:
                    logger.warning(f"Skipping small detection: {crop_width}x{crop_height}")
                    continue
                
                # Crop the object
                cropped_image = image[y1:y2, x1:x2]
                
                # Generate unique filename
                crop_filename = f"crop_{uuid.uuid4().hex}_{i}.jpg"
                crop_path = os.path.join(self.temp_dir, crop_filename)
                
                # Save cropped image
                cv2.imwrite(crop_path, cropped_image)
                
                # Add cropped image info to detection result
                detection_result = detection.copy()
                detection_result.update({
                    'cropped_image_path': crop_path,
                    'cropped_image_array': cropped_image,
                    'crop_bbox': {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': crop_width, 'height': crop_height
                    }
                })
                
                results.append(detection_result)
            
            logger.info(f"Successfully cropped {len(results)} objects from {image_path}")
            return results
            
        except Exception as e:
            logger.error(f"Error cropping detected objects: {str(e)}")
            raise
    
    def crop_from_bytes(self, image_bytes: bytes, detections: List[Dict], 
                       padding: int = 10, min_size: int = 50) -> List[Dict]:
        """
        Crop detected objects from image bytes
        
        Args:
            image_bytes: Image data as bytes
            detections: List of detection results
            padding: Extra padding around bounding box
            min_size: Minimum size for cropped objects
            
        Returns:
            List of detection results with cropped images
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image from bytes")
            
            h, w = image.shape[:2]
            results = []
            
            for i, detection in enumerate(detections):
                bbox = detection['bbox']
                
                # Add padding to bounding box
                x1 = max(0, bbox['x1'] - padding)
                y1 = max(0, bbox['y1'] - padding)
                x2 = min(w, bbox['x2'] + padding)
                y2 = min(h, bbox['y2'] + padding)
                
                # Check minimum size
                crop_width = x2 - x1
                crop_height = y2 - y1
                
                if crop_width < min_size or crop_height < min_size:
                    continue
                
                # Crop the object
                cropped_image = image[y1:y2, x1:x2]
                
                # Convert to bytes
                _, cropped_bytes = cv2.imencode('.jpg', cropped_image)
                cropped_bytes = cropped_bytes.tobytes()
                
                # Add cropped image info to detection result
                detection_result = detection.copy()
                detection_result.update({
                    'cropped_image_bytes': cropped_bytes,
                    'cropped_image_array': cropped_image,
                    'crop_bbox': {
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'width': crop_width, 'height': crop_height
                    }
                })
                
                results.append(detection_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error cropping from bytes: {str(e)}")
            raise
    
    def resize_image(self, image: Union[str, np.ndarray, bytes], 
                    target_size: Tuple[int, int] = (224, 224), 
                    maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize image to target size
        
        Args:
            image: Image path, numpy array, or bytes
            target_size: Target (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image as numpy array
        """
        try:
            # Convert to numpy array
            if isinstance(image, str):
                img_array = cv2.imread(image)
            elif isinstance(image, bytes):
                nparr = np.frombuffer(image, np.uint8)
                img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img_array = image.copy()
            
            if img_array is None:
                raise ValueError("Failed to load image")
            
            if maintain_aspect_ratio:
                # Resize maintaining aspect ratio with padding
                resized = self._resize_with_padding(img_array, target_size)
            else:
                # Direct resize
                resized = cv2.resize(img_array, target_size)
            
            return resized
            
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            raise
    
    def _resize_with_padding(self, image: np.ndarray, 
                           target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image maintaining aspect ratio with padding
        
        Args:
            image: Input image array
            target_size: Target (width, height)
            
        Returns:
            Resized image with padding
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create canvas with target size
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Calculate position to center the image
        start_x = (target_w - new_w) // 2
        start_y = (target_h - new_h) // 2
        
        # Place resized image on canvas
        canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
        
        return canvas
    
    def enhance_image(self, image: np.ndarray, 
                     brightness: float = 1.0, 
                     contrast: float = 1.0, 
                     saturation: float = 1.0) -> np.ndarray:
        """
        Enhance image with brightness, contrast, and saturation adjustments
        
        Args:
            image: Input image array
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            saturation: Saturation factor (1.0 = no change)
            
        Returns:
            Enhanced image array
        """
        try:
            # Convert to PIL for enhancement
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Apply enhancements
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast)
            
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(saturation)
            
            # Convert back to OpenCV format
            enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            raise
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to 0-1 range
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image array
        """
        try:
            normalized = image.astype(np.float32) / 255.0
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing image: {str(e)}")
            raise
    
    def preprocess_for_embedding(self, image: Union[str, np.ndarray, bytes], 
                                target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess image for embedding generation
        
        Args:
            image: Input image
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image array
        """
        try:
            # Resize image
            resized = self.resize_image(image, target_size, maintain_aspect_ratio=True)
            
            # Enhance image quality
            enhanced = self.enhance_image(resized, brightness=1.1, contrast=1.1)
            
            # Normalize
            normalized = self.normalize_image(enhanced)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error preprocessing image for embedding: {str(e)}")
            raise
    
    def save_processed_image(self, image: np.ndarray, 
                           filename: str = None, 
                           format: str = 'jpg') -> str:
        """
        Save processed image to temp directory
        
        Args:
            image: Image array to save
            filename: Optional filename
            format: Image format ('jpg', 'png')
            
        Returns:
            Path to saved image
        """
        try:
            if filename is None:
                filename = f"processed_{uuid.uuid4().hex}.{format}"
            
            filepath = os.path.join(self.temp_dir, filename)
            
            # Save image
            if format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif format.lower() == 'png':
                cv2.imwrite(filepath, image)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Image saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving processed image: {str(e)}")
            raise
    
    def image_to_base64(self, image: np.ndarray, format: str = 'jpg') -> str:
        """
        Convert image array to base64 string
        
        Args:
            image: Image array
            format: Image format ('jpg', 'png')
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Encode image
            if format.lower() in ['jpg', 'jpeg']:
                _, buffer = cv2.imencode('.jpg', image)
            elif format.lower() == 'png':
                _, buffer = cv2.imencode('.png', image)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            raise
    
    def base64_to_image(self, base64_string: str) -> np.ndarray:
        """
        Convert base64 string to image array
        
        Args:
            base64_string: Base64 encoded image
            
        Returns:
            Image array
        """
        try:
            # Decode base64
            image_bytes = base64.b64decode(base64_string)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode base64 image")
            
            return image
            
        except Exception as e:
            logger.error(f"Error converting base64 to image: {str(e)}")
            raise
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Clean up old temporary files
        
        Args:
            max_age_hours: Maximum age of files to keep in hours
        """
        try:
            import time
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)
            
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.isfile(filepath):
                    file_time = os.path.getmtime(filepath)
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Removed old temp file: {filename}")
                        
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {str(e)}")
    
    def get_image_info(self, image: Union[str, np.ndarray, bytes]) -> Dict:
        """
        Get image information
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image info
        """
        try:
            # Convert to numpy array
            if isinstance(image, str):
                img_array = cv2.imread(image)
            elif isinstance(image, bytes):
                nparr = np.frombuffer(image, np.uint8)
                img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img_array = image.copy()
            
            if img_array is None:
                raise ValueError("Failed to load image")
            
            h, w, c = img_array.shape
            
            info = {
                'width': w,
                'height': h,
                'channels': c,
                'dtype': str(img_array.dtype),
                'size_bytes': img_array.nbytes,
                'aspect_ratio': w / h
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting image info: {str(e)}")
            raise


# Singleton instance for global use
_processor_instance = None

def get_processor(temp_dir: str = None) -> FashionImageProcessor:
    """
    Get singleton image processor instance
    
    Args:
        temp_dir: Directory for temporary files
        
    Returns:
        FashionImageProcessor instance
    """
    global _processor_instance
    
    if _processor_instance is None:
        _processor_instance = FashionImageProcessor(temp_dir)
    
    return _processor_instance