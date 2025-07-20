import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class FashionYOLODetector:
    """
    Fashion object detection using YOLOv11 model
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Set default model path if not provided
        if model_path is None:
            current_dir = Path(__file__).parent
            model_path = str(current_dir / "best.pt")
        
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {str(e)}")
            raise
    
    def detect_objects(self, image_path: str) -> List[Dict]:
        """
        Detect fashion objects in image
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detected objects with bounding boxes and class info
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Run YOLO inference
            results = self.model(image_path, conf=self.confidence_threshold)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Debug log for classification
                        print(f"YOLO Detection: {class_name} (confidence: {confidence:.3f})")
                        
                        detection = {
                            'bbox': {
                                'x1': int(x1),
                                'y1': int(y1),
                                'x2': int(x2),
                                'y2': int(y2),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1)
                            },
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'detection_id': f"det_{i}_{class_name}"
                        }
                        
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} objects in {image_path}")
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {str(e)}")
            raise
    
    def detect_from_bytes(self, image_bytes: bytes) -> List[Dict]:
        """
        Detect objects from image bytes
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            List of detected objects
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image from bytes")
            
            # Run YOLO inference on numpy array
            results = self.model(image, conf=self.confidence_threshold)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]
                        
                        # Debug log for classification
                        print(f"YOLO Detection: {class_name} (confidence: {confidence:.3f})")
                        
                        detection = {
                            'bbox': {
                                'x1': int(x1),
                                'y1': int(y1),
                                'x2': int(x2),
                                'y2': int(y2),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1)
                            },
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'detection_id': f"det_{i}_{class_name}"
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in detection from bytes: {str(e)}")
            raise
    
    def crop_detected_objects(self, image_path: str, detections: List[Dict]) -> List[Dict]:
        """
        Crop detected objects from original image
        
        Args:
            image_path: Path to original image
            detections: List of detection results
            
        Returns:
            List of detection results with cropped images
        """
        try:
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            results = []
            
            for detection in detections:
                bbox = detection['bbox']
                
                # Extract bounding box coordinates
                x1, y1 = bbox['x1'], bbox['y1']
                x2, y2 = bbox['x2'], bbox['y2']
                
                # Crop the object
                cropped_image = image[y1:y2, x1:x2]
                
                # Add cropped image to detection result
                detection_result = detection.copy()
                detection_result['cropped_image'] = cropped_image
                
                results.append(detection_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cropping objects: {str(e)}")
            raise
    
    def get_model_classes(self) -> Dict[int, str]:
        """
        Get model class names
        
        Returns:
            Dictionary mapping class IDs to class names
        """
        if self.model is None:
            return {}
        
        return self.model.names
    
    def set_confidence_threshold(self, threshold: float):
        """
        Update confidence threshold
        
        Args:
            threshold: New confidence threshold (0.0 - 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold updated to {threshold}")
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def filter_fashion_objects(self, detections: List[Dict], 
                              fashion_classes: List[str] = None) -> List[Dict]:
        """
        Filter detections to keep only fashion-related objects
        
        Args:
            detections: List of all detections
            fashion_classes: List of fashion class names to keep
            
        Returns:
            Filtered list of fashion objects
        """
        if fashion_classes is None:
            # Default fashion classes - adjust based on your trained model
            fashion_classes = [
                'person', 'shirt', 'pants', 'dress', 'jacket', 'coat', 'sweater', 'suit',
                'shoe', 'boot', 'sneaker', 'sandal', 'heel',
                'hat', 'cap', 'bag', 'purse', 'backpack', 'handbag',
                'necklace', 'bracelet', 'ring', 'earring', 'watch',
                'belt', 'scarf', 'glasses', 'sunglasses', 'tie'
            ]
        
        fashion_detections = []
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            
            # Check if detected class is in fashion classes
            if any(fashion_class in class_name for fashion_class in fashion_classes):
                fashion_detections.append(detection)
        
        logger.info(f"Filtered {len(fashion_detections)} fashion objects from {len(detections)} total detections")
        return fashion_detections


# Singleton instance for global use
_detector_instance = None

def get_detector(model_path: str = None, confidence_threshold: float = 0.5) -> FashionYOLODetector:
    """
    Get singleton YOLO detector instance
    
    Args:
        model_path: Path to YOLO model file
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        FashionYOLODetector instance
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = FashionYOLODetector(model_path, confidence_threshold)
    
    return _detector_instance