#!/usr/bin/env python
"""
Test script for batch processor
"""
import sys
import os

# Add backend app to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'app')
sys.path.insert(0, backend_path)

from backend.app.services.batch_processor import FashionBatchProcessor

def main():
    # Get first 5 image files for testing
    train_dir = '/home/developer/Desktop/images/images/train'
    image_files = []
    for f in os.listdir(train_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(os.path.join(train_dir, f))
        # if len(image_files) >= 5:  # Process only first 5 for testing
        #     break
    
    print(f'Testing batch processing with {len(image_files)} images:')
    for img in image_files:
        print(f'  - {os.path.basename(img)}')
    
    # Initialize processor
    processor = FashionBatchProcessor(confidence_threshold=0.3)
    
    # Process images one by one to see progress
    total_objects = 0
    successful_images = 0
    
    for i, image_path in enumerate(image_files):
        print(f'\n--- Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)} ---')
        try:
            result = processor.process_single_image(image_path)
            if result['success']:
                successful_images += 1
                objects_count = result.get('objects_processed', 0)
                total_objects += objects_count
                print(f'✓ Success: {objects_count} objects processed')
                
                # Show categories detected
                if result.get('categories'):
                    print(f'  Categories: {", ".join(result["categories"])}')
                
                # Show MinIO URL
                if result.get('minio_url'):
                    print(f'  MinIO: {result["minio_url"]}')
                    
            else:
                print(f'✗ Failed: {result.get("error", "Unknown error")}')
                
        except Exception as e:
            print(f'✗ Exception: {str(e)}')
    
    print(f'\n=== BATCH PROCESSING SUMMARY ===')
    print(f'Total images processed: {successful_images}/{len(image_files)}')
    print(f'Total objects detected and stored: {total_objects}')
    print(f'Success rate: {successful_images/len(image_files)*100:.1f}%')

if __name__ == "__main__":
    main()