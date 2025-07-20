import os
import sys
import json
import cv2
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clip_embedder import get_embedder
from models.yolo_detector import get_detector
from services.qdrant_client import get_qdrant_client
from services.minio_client import get_minio_client

logger = logging.getLogger(__name__)

class FashionBatchProcessor:
    """
    Batch processor for training data with YOLO detection and CLIP embeddings
    """
    
    def __init__(self, confidence_threshold: float = 0.3):
        """
        Initialize batch processor
        
        Args:
            confidence_threshold: YOLO detection confidence threshold
        """
        self.confidence_threshold = confidence_threshold
        self.embedder = None
        self.detector = None
        self.qdrant_client = None
        self.minio_client = None
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all required services"""
        try:
            print("Initializing services...")
            self.embedder = get_embedder()
            self.detector = get_detector(confidence_threshold=self.confidence_threshold)
            self.qdrant_client = get_qdrant_client()
            self.minio_client = get_minio_client()
            print("All services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize services: {str(e)}")
            raise
    
    def process_directory(self, 
                         directory_path: str,
                         metadata_file: Optional[str] = None,
                         recursive: bool = True) -> Dict:
        """
        Process all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            metadata_file: Optional JSON file with metadata for images
            recursive: Process subdirectories recursively
            
        Returns:
            Processing results dictionary
        """
        try:
            # Load metadata if provided
            metadata_dict = {}
            if metadata_file and os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_dict = json.load(f)
            
            # Find all image files
            image_files = self._find_image_files(directory_path, recursive)
            print(f"Found {len(image_files)} image files in {directory_path}")
            
            if not image_files:
                return {
                    'success': False,
                    'message': 'No image files found',
                    'processed_items': []
                }
            
            # Process each image
            processed_items = []
            total_objects = 0
            
            for i, image_path in enumerate(image_files):
                try:
                    print(f"\nProcessing {i+1}/{len(image_files)}: {image_path}")
                    
                    # Get metadata for this image
                    image_name = os.path.basename(image_path)
                    image_metadata = metadata_dict.get(image_name, {})
                    
                    # Process single image
                    result = self.process_single_image(image_path, image_metadata)
                    
                    if result['success']:
                        processed_items.append(result)
                        total_objects += result.get('objects_processed', 0)
                        print(f"✓ Processed {result.get('objects_processed', 0)} objects")
                    else:
                        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    continue
            
            return {
                'success': True,
                'message': f'Processed {len(processed_items)} images with {total_objects} objects',
                'processed_items': processed_items,
                'total_images': len(processed_items),
                'total_objects': total_objects,
                'failed_images': len(image_files) - len(processed_items)
            }
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processed_items': []
            }
    
    def process_single_image(self, 
                            image_path: str, 
                            metadata: Dict = None) -> Dict:
        """
        Process a single image file
        
        Args:
            image_path: Path to image file
            metadata: Optional metadata dictionary
            
        Returns:
            Processing result dictionary
        """
        try:
            if metadata is None:
                metadata = {}
            
            # Generate unique ID
            file_id = str(uuid.uuid4())
            filename = os.path.basename(image_path)
            
            # Upload original image to MinIO
            s3_key = f"training/{file_id}_{filename}"
            
            # Upload to MinIO training bucket
            upload_result = self.minio_client.upload_file(
                file_path=image_path,
                bucket_type='training_data',
                object_name=s3_key,
                metadata={
                    'original_filename': filename,
                    'file_id': file_id,
                    'upload_type': 'training_batch_process',
                    'uploaded_at': datetime.utcnow().isoformat()
                }
            )
            
            if not upload_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to upload image to MinIO',
                    'file_id': file_id,
                    'filename': filename
                }
            
            print(f"Uploaded to MinIO: {upload_result['file_url']}")
            
            # Run YOLO detection
            print("Running YOLO detection...")
            detections = self.detector.detect_objects(image_path)
            fashion_objects = self.detector.filter_fashion_objects(detections)
            print(f"Found {len(fashion_objects)} fashion objects")
            
            if not fashion_objects:
                return {
                    'success': False,
                    'error': 'No fashion objects found',
                    'file_id': file_id,
                    'filename': filename
                }
            
            # Load image for cropping
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to load image',
                    'file_id': file_id,
                    'filename': filename
                }
            
            # Process each detected object
            objects_processed = 0
            object_results = []
            
            for obj_idx, obj in enumerate(fashion_objects):
                try:
                    bbox = obj['bbox']
                    category = obj['class_name']
                    confidence = obj['confidence']
                    
                    # Extract crop coordinates
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    
                    # Crop the object from image
                    cropped_image = image[y1:y2, x1:x2]
                    
                    if cropped_image.size == 0:
                        print(f"Empty crop for object {obj_idx}, skipping...")
                        continue
                    
                    # Save cropped image to MinIO
                    crop_filename = None
                    crop_minio_url = None
                    try:
                        # Convert cropped image to bytes
                        _, buffer = cv2.imencode('.jpg', cropped_image)
                        crop_bytes = buffer.tobytes()
                        
                        # Generate unique filename for crop
                        crop_filename = f"training_{file_id}_crop_{obj_idx}_{category}.jpg"
                        
                        # Upload to cropped_objects bucket
                        crop_result = self.minio_client.upload_bytes(
                            file_bytes=crop_bytes,
                            bucket_type='cropped_objects',
                            object_name=crop_filename,
                            content_type='image/jpeg',
                            metadata={
                                'original_file_id': file_id,
                                'object_index': str(obj_idx),
                                'category': category,
                                'confidence': str(confidence),
                                'bbox_x1': str(x1),
                                'bbox_y1': str(y1),
                                'bbox_x2': str(x2),
                                'bbox_y2': str(y2),
                                'original_filename': filename,
                                'source': 'training_batch_process',
                                'created_at': datetime.utcnow().isoformat()
                            }
                        )
                        crop_minio_url = crop_result['file_url']
                        print(f"Saved cropped image: {crop_filename}")
                    except Exception as crop_error:
                        print(f"Error saving cropped image: {str(crop_error)}")
                    
                    # Generate CLIP embedding from cropped region
                    embedding = self.embedder.encode_image(cropped_image)
                    print(f"Generated embedding for {category}: {embedding.shape}")
                    
                    # Prepare metadata for Qdrant
                    object_id = str(uuid.uuid4())
                    qdrant_metadata = {
                        'product_id': metadata.get('product_id', file_id),
                        'object_id': object_id,
                        'original_object_id': f"{file_id}_{obj_idx}",
                        'category': category,
                        'subcategory': metadata.get('subcategory'),
                        'brand': metadata.get('brand'),
                        'color': metadata.get('color'),
                        'material': metadata.get('material'),
                        'size': metadata.get('size'),
                        'price': metadata.get('price'),
                        'description': metadata.get('description'),
                        'original_image_path': image_path,
                        'original_image_s3_key': s3_key,
                        'original_image_minio_url': upload_result['file_url'],
                        'crop_filename': crop_filename,
                        'crop_minio_url': crop_minio_url,
                        'crop_coordinates': {
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'width': x2-x1, 'height': y2-y1
                        },
                        'detection_confidence': confidence,
                        'embedding_source': 'cropped_region',
                        'type': 'training_data',
                        'processed_at': datetime.utcnow().isoformat()
                    }
                    
                    # Store in Qdrant
                    point_id = self.qdrant_client.store_embedding(
                        embedding=embedding,
                        metadata=qdrant_metadata,
                        collection_type='training_embeddings',
                        point_id=object_id
                    )
                    print(f"Stored object {category} in Qdrant with ID: {point_id}")
                    
                    object_results.append({
                        'object_id': object_id,
                        'category': category,
                        'confidence': confidence,
                        'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                        'crop_filename': crop_filename,
                        'crop_minio_url': crop_minio_url,
                        'qdrant_id': point_id
                    })
                    
                    objects_processed += 1
                    
                except Exception as obj_error:
                    print(f"Error processing object {obj_idx}: {str(obj_error)}")
                    continue
            
            return {
                'success': True,
                'file_id': file_id,
                'filename': filename,
                'image_path': image_path,
                's3_key': s3_key,
                'minio_url': upload_result['file_url'],
                'minio_bucket': upload_result['bucket'],
                'objects_found': len(fashion_objects),
                'objects_processed': objects_processed,
                'object_results': object_results,
                'categories': [obj['class_name'] for obj in fashion_objects]
            }
            
        except Exception as e:
            logger.error(f"Error processing single image {image_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_id': file_id if 'file_id' in locals() else None,
                'filename': filename if 'filename' in locals() else None
            }
    
    def _find_image_files(self, directory_path: str, recursive: bool = True) -> List[str]:
        """Find all image files in directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        image_files = []
        
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        # Get files based on recursive flag
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
        
        return sorted(image_files)
    
    def create_metadata_template(self, directory_path: str, output_file: str):
        """Create a metadata template JSON file for images in directory"""
        try:
            image_files = self._find_image_files(directory_path, recursive=False)
            
            metadata_template = {}
            for image_file in image_files:
                filename = os.path.basename(image_file)
                metadata_template[filename] = {
                    "product_id": "",
                    "category": "",
                    "subcategory": "",
                    "brand": "",
                    "color": "",
                    "material": "",
                    "size": "",
                    "price": None,
                    "description": ""
                }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_template, f, indent=2, ensure_ascii=False)
            
            print(f"Created metadata template with {len(image_files)} entries: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating metadata template: {str(e)}")
            raise


def main():
    """CLI interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fashion Training Data Batch Processor')
    parser.add_argument('directory', help='Directory containing training images')
    parser.add_argument('--metadata', help='JSON file with image metadata')
    parser.add_argument('--recursive', action='store_true', 
                       help='Process subdirectories recursively')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='YOLO detection confidence threshold')
    parser.add_argument('--create-template', help='Create metadata template file')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = FashionBatchProcessor(confidence_threshold=args.confidence)
    
    # Create metadata template if requested
    if args.create_template:
        processor.create_metadata_template(args.directory, args.create_template)
        print(f"Metadata template created: {args.create_template}")
        print("Fill in the metadata and run again with --metadata parameter")
        return
    
    # Process directory
    print(f"Starting batch processing of: {args.directory}")
    results = processor.process_directory(
        directory_path=args.directory,
        metadata_file=args.metadata,
        recursive=args.recursive
    )
    
    # Print results
    if results['success']:
        print(f"\n✓ Batch processing completed successfully!")
        print(f"📊 Summary:")
        print(f"   - Total images processed: {results['total_images']}")
        print(f"   - Total objects found: {results['total_objects']}")
        print(f"   - Failed images: {results.get('failed_images', 0)}")
    else:
        print(f"\n✗ Batch processing failed: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()