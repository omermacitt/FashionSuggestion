from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError
import cv2
import json

app = Flask(__name__, 
           template_folder='../../frontend/templates',
           static_folder='../../frontend/static')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

from dotenv import load_dotenv
load_dotenv()

# S3/MinIO Configuration
S3_ENDPOINT = os.getenv('S3_ENDPOINT', 'http://localhost:9000')
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'fashion-uploads')

# Initialize S3 client
s3_client = boto3.client(
    's3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    region_name='us-east-1'
)

# Ensure bucket exists
try:
    s3_client.head_bucket(Bucket=S3_BUCKET_NAME)
except ClientError:
    try:
        s3_client.create_bucket(Bucket=S3_BUCKET_NAME)
    except ClientError as e:
        print(f"Error creating bucket: {e}")

# Store results in memory (in production, use a database)
results_store = {}

@app.route('/')
def index():
    """Ana sayfa - görsel upload"""
    return render_template('index.html')

@app.route('/results')
def results():
    """Sonuçlar sayfası"""
    return render_template('results.html')

@app.route('/api/load-file', methods=['POST'])
def load_file():
    """Dosya yükleme endpoint'i"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{file_id}.{file_extension}"
        
        try:
            # Upload to S3/MinIO
            s3_client.upload_fileobj(
                file,
                S3_BUCKET_NAME,
                unique_filename,
                ExtraArgs={'ContentType': file.content_type}
            )
            
            # Store initial result data
            results_store[file_id] = {
                'file_id': file_id,
                'filename': unique_filename,
                'original_filename': filename,
                'timestamp': datetime.now().isoformat(),
                'status': 'uploaded',
                'suggestions': []
            }
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'filename': unique_filename,
                'message': 'File uploaded successfully to S3'
            })
            
        except ClientError as e:
            return jsonify({'error': f'Failed to upload file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/result/<file_id>', methods=['GET'])
def get_result(file_id):
    """Belirli bir dosya için sonuç döndürme"""
    if file_id not in results_store:
        return jsonify({'error': 'Result not found'}), 404
    
    return jsonify(results_store[file_id])

@app.route('/api/image-proxy/<path:bucket>/<path:object_path>', methods=['GET'])
def image_proxy(bucket, object_path):
    """MinIO image proxy for frontend access"""
    try:
        import tempfile
        import os
        from flask import send_file
        
        # Download from MinIO
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            # Create a temporary MinIO client for downloading
            minio_s3 = boto3.client(
                's3',
                endpoint_url=S3_ENDPOINT,
                aws_access_key_id=S3_ACCESS_KEY,
                aws_secret_access_key=S3_SECRET_KEY
            )
            minio_s3.download_file(bucket, object_path, tmp_file.name)
            
            # Send file to frontend
            return send_file(tmp_file.name, mimetype='image/jpeg', as_attachment=False)
            
    except Exception as e:
        print(f"Image proxy error: {str(e)}")
        return "Image not found", 404

@app.route('/api/results/history', methods=['GET'])
def get_history():
    """Geçmiş sonuçları döndürme"""
    history = []
    for file_id, result in results_store.items():
        history.append({
            'file_id': file_id,
            'timestamp': result.get('timestamp'),
            'filename': result.get('filename'),
            'suggestions_count': len(result.get('suggestions', []))
        })
    
    # Sort by timestamp (newest first)
    history.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({'history': history})

@app.route('/api/detect', methods=['POST'])
def detect_objects():
    """Object detection endpoint"""
    data = request.get_json()
    file_id = data.get('file_id')
    
    if not file_id or file_id not in results_store:
        return jsonify({'error': 'File not found'}), 404
    
    try:
        print(f"Starting detection for file_id: {file_id}")
        
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from models.yolo_detector import get_detector
            print("YOLO detector imported successfully")
        except Exception as import_err:
            print(f"Import error: {import_err}")
            raise import_err
        
        # Get file from S3
        filename = results_store[file_id]['filename']
        print(f"Downloading file: {filename}")
        
        # Download file from S3 to temporary location for processing
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                s3_client.download_file(S3_BUCKET_NAME, filename, tmp_file.name)
                temp_path = tmp_file.name
                print(f"File downloaded to: {temp_path}")
        except Exception as download_err:
            print(f"S3 download error: {download_err}")
            raise download_err
        
        try:
            # Initialize YOLO detector
            print("Initializing YOLO detector...")
            detector = get_detector(confidence_threshold=0.3)
            print("YOLO detector initialized")
            
            # Detect objects
            print("Running object detection...")
            detections = detector.detect_objects(temp_path)
            print(f"Raw detections: {len(detections)}")
            
            # Debug: Print all raw detections
            for i, det in enumerate(detections):
                print(f"  Raw {i}: {det['class_name']} (conf: {det['confidence']:.3f})")
            
            # Filter for fashion objects only
            fashion_objects = detector.filter_fashion_objects(detections)
            print(f"Fashion objects: {len(fashion_objects)}")
            
            # Debug: Print filtered fashion objects
            for i, obj in enumerate(fashion_objects):
                print(f"  Fashion {i}: {obj['class_name']} (conf: {obj['confidence']:.3f})")
            
            # Convert to API format
            detected_objects = []
            for obj in fashion_objects:
                detected_objects.append({
                    'class': obj['class_name'],
                    'confidence': obj['confidence'],
                    'bbox': [
                        obj['bbox']['x1'], 
                        obj['bbox']['y1'], 
                        obj['bbox']['x2'], 
                        obj['bbox']['y2']
                    ],
                    'detection_id': obj['detection_id']
                })
            
            # Update results store
            results_store[file_id]['detected_objects'] = detected_objects
            results_store[file_id]['status'] = 'detected'
            
            print(f"Detection successful: {len(detected_objects)} objects")
            return jsonify({
                'success': True,
                'detected_objects': detected_objects,
                'message': f'Detected {len(detected_objects)} fashion objects'
            })
            
        finally:
            # Clean up temporary file
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
                print("Temporary file cleaned up")
        
    except Exception as e:
        print(f"Detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/api/search', methods=['POST'])
def search_similar():
    """Similar product search endpoint using real CLIP embeddings"""
    data = request.get_json()
    file_id = data.get('file_id')
    
    if not file_id or file_id not in results_store:
        return jsonify({'error': 'File not found'}), 404
    
    try:
        print(f"Starting similarity search for file_id: {file_id}")
        
        # Import required modules
        import sys
        import tempfile
        import cv2
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from models.clip_embedder import get_embedder
        from services.qdrant_client import get_qdrant_client
        
        # Initialize services
        embedder = get_embedder()
        qdrant_client = get_qdrant_client()
        
        # Get detected objects from the detection step
        detected_objects = results_store[file_id].get('detected_objects', [])
        print(f"DEBUG: Found {len(detected_objects)} detected objects:")
        for i, obj in enumerate(detected_objects):
            print(f"  Object {i}: {obj.get('class', 'unknown')} (confidence: {obj.get('confidence', 0):.3f})")
        
        if not detected_objects:
            return jsonify({
                'success': True,
                'data': [],
                'message': 'No objects detected to search for'
            })
        
        # Get original file info
        filename = results_store[file_id]['filename']
        
        # Download file from S3 for processing
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            s3_client.download_file(S3_BUCKET_NAME, filename, tmp_file.name)
            temp_path = tmp_file.name
        
        try:
            # Load the image
            image = cv2.imread(temp_path)
            if image is None:
                return jsonify({'error': 'Failed to load image for search'}), 500
            
            search_results = []
            
            # Process each detected object
            for obj_idx, obj in enumerate(detected_objects):
                try:
                    category = obj['class']
                    confidence = obj['confidence']
                    bbox = obj['bbox']  # [x1, y1, x2, y2]
                    
                    print(f"Searching for {category} with confidence {confidence:.2f}")
                    
                    # Extract coordinates
                    x1, y1, x2, y2 = bbox
                    
                    # Crop the detected object
                    cropped_image = image[y1:y2, x1:x2]
                    
                    if cropped_image.size == 0:
                        print(f"Empty crop for object {obj_idx}, skipping...")
                        continue
                    
                    # Save cropped image to MinIO for debugging
                    crop_filename = None
                    try:
                        from services.minio_client import get_minio_client
                        minio_client = get_minio_client()
                        
                        # Convert cropped image to bytes
                        _, buffer = cv2.imencode('.jpg', cropped_image)
                        crop_bytes = buffer.tobytes()
                        
                        # Generate unique filename for crop
                        crop_filename = f"{file_id}_crop_{obj_idx}_{category}.jpg"
                        
                        # Upload to cropped_objects bucket
                        crop_result = minio_client.upload_bytes(
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
                                'bbox_y2': str(y2)
                            }
                        )
                        print(f"Saved cropped image: {crop_filename}")
                    except Exception as crop_error:
                        print(f"Error saving cropped image: {str(crop_error)}")
                    
                    # Generate CLIP embedding for the cropped object
                    query_embedding = embedder.encode_image(cropped_image)
                    print(f"Generated query embedding: {query_embedding.shape}")
                    
                    # Search in Qdrant for similar objects
                    # Handle suit category fallback to pants (since training data has pants but no suits)
                    search_category = category
                    if category == 'suit' and True:  # Always fallback for now
                        search_category = 'pants'
                        print(f"No suit data found, searching in pants category instead")
                    
                    similar_items = qdrant_client.search_similar(
                        query_embedding=query_embedding,
                        collection_type='training_embeddings',
                        limit=5,
                        score_threshold=0.6,
                        filters={'category': search_category}
                    )
                    
                    print(f"Found {len(similar_items)} similar items for {category}")
                    
                    # Format similar products
                    similar_products = []
                    for item in similar_items:
                        metadata = item['metadata']
                        
                        # Get image URL via backend proxy
                        s3_key = metadata.get('original_image_s3_key', '')
                        bucket_name = 'training-data' if 'training/' in s3_key else S3_BUCKET_NAME
                        image_url = f"http://localhost:3000/api/image-proxy/{bucket_name}/{s3_key}"
                        
                        similar_products.append({
                            'image_url': image_url,
                            'similarity': round(item['score'], 3),
                            'product_name': metadata.get('description', f"{metadata.get('brand', 'Unknown')} {category}"),
                            'price': f"{metadata.get('price', 'N/A')} TL" if metadata.get('price') else 'Fiyat Yok',
                            'brand': metadata.get('brand', 'Unknown'),
                            'color': metadata.get('color', 'Unknown'),
                            'category': metadata.get('category', category),
                            'product_id': metadata.get('product_id'),
                            'crop_coordinates': metadata.get('crop_coordinates', {})
                        })
                    
                    if similar_products:
                        search_results.append({
                            'object_type': category,
                            'detection_confidence': confidence,
                            'bbox': bbox,
                            'crop_filename': crop_filename,
                            'crop_url': f"http://localhost:3000/api/image-proxy/cropped-objects/{crop_filename}" if crop_filename else None,
                            'similar_products': similar_products
                        })
                    
                except Exception as obj_error:
                    print(f"Error processing object {obj_idx}: {str(obj_error)}")
                    continue
            
            # Update results store
            results_store[file_id]['suggestions'] = search_results
            results_store[file_id]['status'] = 'completed'
            
            print(f"Search completed successfully. Found {len(search_results)} result groups with total similar products:")
            for i, result in enumerate(search_results):
                print(f"  Group {i+1}: {result['object_type']} with {len(result['similar_products'])} similar items")
            
            return jsonify({
                'success': True,
                'data': search_results,
                'message': f'Found similar products for {len(search_results)} objects'
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/api/add-training-data', methods=['POST'])
def add_training_data():
    """Add training data to S3 and Qdrant with object detection and cropping"""
    try:
        # Check if files are present
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        metadata_json = request.form.get('metadata', '[]')
        
        try:
            metadata_list = json.loads(metadata_json)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid metadata JSON'}), 400
        
        if len(files) != len(metadata_list):
            return jsonify({'error': 'Number of images and metadata must match'}), 400
        
        print(f"Processing {len(files)} training images...")
        
        # Import required modules
        import sys
        import tempfile
        import json
        import cv2
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from models.clip_embedder import get_embedder
        from models.yolo_detector import get_detector
        from services.qdrant_client import get_qdrant_client
        
        # Initialize services
        embedder = get_embedder()
        detector = get_detector(confidence_threshold=0.3)
        qdrant_client = get_qdrant_client()
        
        processed_items = []
        total_objects_found = 0
        
        for i, (file, metadata) in enumerate(zip(files, metadata_list)):
            try:
                print(f"Processing image {i+1}/{len(files)}: {file.filename}")
                
                # Validate file
                if not allowed_file(file.filename):
                    print(f"Skipping invalid file: {file.filename}")
                    continue
                
                # Generate unique filename for S3 (original image)
                file_id = str(uuid.uuid4())
                filename = secure_filename(file.filename)
                file_extension = filename.rsplit('.', 1)[1].lower()
                unique_filename = f"training/{file_id}.{file_extension}"
                
                # Upload original image to S3
                file.seek(0)  # Reset file pointer
                s3_client.upload_fileobj(
                    file,
                    S3_BUCKET_NAME,
                    unique_filename,
                    ExtraArgs={'ContentType': file.content_type}
                )
                print(f"Uploaded original to S3: {unique_filename}")
                
                # Process image with temporary file
                with tempfile.NamedTemporaryFile(suffix=f'.{file_extension}', delete=False) as tmp_file:
                    file.seek(0)
                    tmp_file.write(file.read())
                    temp_path = tmp_file.name
                
                try:
                    # 1. Run YOLO detection on original image
                    print("Running YOLO detection...")
                    detections = detector.detect_objects(temp_path)
                    fashion_objects = detector.filter_fashion_objects(detections)
                    print(f"Found {len(fashion_objects)} fashion objects")
                    
                    if not fashion_objects:
                        print(f"No fashion objects found in {filename}, skipping...")
                        continue
                    
                    # 2. Load image for cropping
                    image = cv2.imread(temp_path)
                    if image is None:
                        print(f"Failed to load image: {filename}")
                        continue
                    
                    # 3. Process each detected object
                    objects_processed = 0
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
                                from services.minio_client import get_minio_client
                                minio_client = get_minio_client()
                                
                                # Convert cropped image to bytes
                                _, buffer = cv2.imencode('.jpg', cropped_image)
                                crop_bytes = buffer.tobytes()
                                
                                # Generate unique filename for crop
                                crop_filename = f"training_{file_id}_crop_{obj_idx}_{category}.jpg"
                                
                                # Upload to cropped_objects bucket
                                crop_result = minio_client.upload_bytes(
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
                                        'source': 'training_data_upload',
                                        'created_at': datetime.utcnow().isoformat()
                                    }
                                )
                                crop_minio_url = crop_result['file_url']
                                print(f"Saved cropped training image: {crop_filename}")
                            except Exception as crop_error:
                                print(f"Error saving cropped training image: {str(crop_error)}")
                            
                            # 4. Generate CLIP embedding from cropped region
                            embedding = embedder.encode_image(cropped_image)
                            print(f"Generated embedding for {category}: {embedding.shape}")
                            
                            # 5. Prepare metadata for Qdrant
                            object_id = f"{file_id}_{obj_idx}"
                            qdrant_metadata = {
                                'product_id': metadata.get('product_id', file_id),
                                'object_id': object_id,
                                'category': category,
                                'subcategory': metadata.get('subcategory'),
                                'brand': metadata.get('brand'),
                                'color': metadata.get('color'),
                                'material': metadata.get('material'),
                                'size': metadata.get('size'),
                                'price': metadata.get('price'),
                                'description': metadata.get('description'),
                                'original_image_s3_key': unique_filename,
                                'crop_filename': crop_filename,
                                'crop_minio_url': crop_minio_url,
                                'crop_coordinates': {
                                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                                    'width': x2-x1, 'height': y2-y1
                                },
                                'detection_confidence': confidence,
                                'embedding_source': 'cropped_region',
                                'type': 'training_data'
                            }
                            
                            # 6. Store in Qdrant
                            point_id = qdrant_client.store_embedding(
                                embedding=embedding,
                                metadata=qdrant_metadata,
                                collection_type='training_embeddings',
                                point_id=object_id
                            )
                            print(f"Stored object {category} in Qdrant with ID: {point_id}")
                            
                            objects_processed += 1
                            total_objects_found += 1
                            
                        except Exception as obj_error:
                            print(f"Error processing object {obj_idx}: {str(obj_error)}")
                            continue
                    
                    processed_items.append({
                        'file_id': file_id,
                        'filename': filename,
                        's3_key': unique_filename,
                        'objects_found': len(fashion_objects),
                        'objects_processed': objects_processed,
                        'categories': [obj['class_name'] for obj in fashion_objects]
                    })
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                
            except Exception as item_error:
                print(f"Error processing item {i}: {str(item_error)}")
                continue
        
        return jsonify({
            'success': True,
            'message': f'Successfully processed {len(processed_items)} images with {total_objects_found} objects',
            'processed_items': processed_items,
            'total_images': len(processed_items),
            'total_objects': total_objects_found
        })
        
    except Exception as e:
        print(f"Training data upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to add training data: {str(e)}'}), 500

@app.route('/api/debug/cropped-images/<file_id>', methods=['GET'])
def get_cropped_images(file_id):
    """Get all cropped images for a specific file_id"""
    try:
        from services.minio_client import get_minio_client
        minio_client = get_minio_client()
        
        # List all cropped images for this file_id
        cropped_files = minio_client.list_files(
            bucket_type='cropped_objects',
            prefix=f"{file_id}_crop_"
        )
        
        # Format response
        cropped_images = []
        for file_info in cropped_files:
            object_name = file_info['object_name']
            
            # Get file metadata
            try:
                file_details = minio_client.get_file_info('cropped_objects', object_name)
                metadata = file_details.get('metadata', {})
                
                cropped_images.append({
                    'object_name': object_name,
                    'image_url': f"http://localhost:3000/api/image-proxy/cropped-objects/{object_name}",
                    'category': metadata.get('category', 'unknown'),
                    'confidence': float(metadata.get('confidence', 0)),
                    'object_index': int(metadata.get('object_index', 0)),
                    'bbox': {
                        'x1': int(metadata.get('bbox_x1', 0)),
                        'y1': int(metadata.get('bbox_y1', 0)),
                        'x2': int(metadata.get('bbox_x2', 0)),
                        'y2': int(metadata.get('bbox_y2', 0))
                    },
                    'size': file_info['size'],
                    'created': file_info['last_modified']
                })
            except Exception as meta_error:
                print(f"Error getting metadata for {object_name}: {str(meta_error)}")
                cropped_images.append({
                    'object_name': object_name,
                    'image_url': f"http://localhost:3000/api/image-proxy/cropped-objects/{object_name}",
                    'size': file_info['size'],
                    'created': file_info['last_modified']
                })
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'cropped_images': cropped_images,
            'count': len(cropped_images)
        })
        
    except Exception as e:
        print(f"Error getting cropped images: {str(e)}")
        return jsonify({'error': f'Failed to get cropped images: {str(e)}'}), 500

@app.route('/api/debug/similarity-analysis', methods=['POST'])
def analyze_similarity():
    """Analyze similarity scores between query and results in detail"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        object_index = data.get('object_index', 0)
        
        if not file_id:
            return jsonify({'error': 'file_id is required'}), 400
        
        # Import required modules
        import sys
        import tempfile
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from models.clip_embedder import get_embedder
        from services.qdrant_client import get_qdrant_client
        from services.minio_client import get_minio_client
        
        # Initialize services
        embedder = get_embedder()
        qdrant_client = get_qdrant_client()
        minio_client = get_minio_client()
        
        # Get the cropped image for analysis
        crop_files = minio_client.list_files(
            bucket_type='cropped_objects',
            prefix=f"{file_id}_crop_{object_index}_"
        )
        
        if not crop_files:
            return jsonify({'error': 'Cropped image not found'}), 404
        
        crop_filename = crop_files[0]['object_name']
        
        # Download and process cropped image
        crop_bytes = minio_client.download_bytes('cropped_objects', crop_filename)
        
        # Convert bytes to numpy array for CLIP
        import numpy as np
        nparr = np.frombuffer(crop_bytes, np.uint8)
        crop_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Generate query embedding
        query_embedding = embedder.encode_image(crop_image)
        
        # Get crop metadata to determine category
        crop_info = minio_client.get_file_info('cropped_objects', crop_filename)
        crop_metadata = crop_info.get('metadata', {})
        category = crop_metadata.get('category', 'unknown')
        
        # Search with different score thresholds
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        analysis_results = []
        
        for threshold in thresholds:
            similar_items = qdrant_client.search_similar(
                query_embedding=query_embedding,
                collection_type='training_embeddings',
                limit=10,
                score_threshold=threshold,
                filters={'category': category}
            )
            
            analysis_results.append({
                'threshold': threshold,
                'results_count': len(similar_items),
                'top_scores': [round(item['score'], 3) for item in similar_items[:5]],
                'avg_score': round(sum(item['score'] for item in similar_items) / len(similar_items), 3) if similar_items else 0
            })
        
        # Also try without category filter
        no_filter_results = qdrant_client.search_similar(
            query_embedding=query_embedding,
            collection_type='training_embeddings',
            limit=10,
            score_threshold=0.3
        )
        
        return jsonify({
            'success': True,
            'crop_info': {
                'filename': crop_filename,
                'category': category,
                'image_url': f"http://localhost:3000/api/image-proxy/cropped-objects/{crop_filename}"
            },
            'embedding_info': {
                'shape': query_embedding.shape,
                'norm': float(np.linalg.norm(query_embedding))
            },
            'threshold_analysis': analysis_results,
            'no_filter_results': {
                'count': len(no_filter_results),
                'top_scores': [round(item['score'], 3) for item in no_filter_results[:10]],
                'categories': list(set(item['metadata'].get('category', 'unknown') for item in no_filter_results))
            }
        })
        
    except Exception as e:
        print(f"Similarity analysis error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Similarity analysis failed: {str(e)}'}), 500

@app.route('/api/debug/training-data-stats', methods=['GET'])
def get_training_data_stats():
    """Get statistics about training data in Qdrant"""
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from services.qdrant_client import get_qdrant_client
        
        qdrant_client = get_qdrant_client()
        
        # Get collection info
        collection_info = qdrant_client.get_collection_info('training_embeddings')
        
        # Count by categories
        categories = ['pants', 'shirt', 'dress', 'jacket', 'shoe', 'bag', 'hat']
        category_counts = {}
        
        for category in categories:
            count = qdrant_client.count_embeddings(
                collection_type='training_embeddings',
                filters={'category': category}
            )
            category_counts[category] = count
        
        # Total count
        total_count = qdrant_client.count_embeddings('training_embeddings')
        
        return jsonify({
            'success': True,
            'collection_info': collection_info,
            'total_embeddings': total_count,
            'category_counts': category_counts,
            'categories_with_data': [cat for cat, count in category_counts.items() if count > 0]
        })
        
    except Exception as e:
        print(f"Training data stats error: {str(e)}")
        return jsonify({'error': f'Failed to get training data stats: {str(e)}'}), 500

@app.route('/api/admin/cleanup-all', methods=['POST'])
def cleanup_all_data():
    """Clean up all MinIO buckets and Qdrant collections"""
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from services.qdrant_client import get_qdrant_client
        from services.minio_client import get_minio_client
        
        qdrant_client = get_qdrant_client()
        minio_client = get_minio_client()
        
        cleanup_results = {
            'qdrant_cleanup': {},
            'minio_cleanup': {},
            'success': True,
            'errors': []
        }
        
        # Clean Qdrant collections
        try:
            for collection_name in ['fashion-embeddings', 'training-embeddings', 'user-embeddings']:
                try:
                    # Delete collection
                    qdrant_client.client.delete_collection(collection_name)
                    print(f"Deleted Qdrant collection: {collection_name}")
                    cleanup_results['qdrant_cleanup'][collection_name] = 'deleted'
                except Exception as e:
                    print(f"Error deleting collection {collection_name}: {str(e)}")
                    cleanup_results['qdrant_cleanup'][collection_name] = f'error: {str(e)}'
            
            # Recreate collections
            qdrant_client._ensure_collections()
            print("Recreated Qdrant collections")
            
        except Exception as e:
            cleanup_results['errors'].append(f"Qdrant cleanup error: {str(e)}")
            cleanup_results['success'] = False
        
        # Clean MinIO buckets
        try:
            for bucket_type in ['user_uploads', 'training_data', 'cropped_objects', 'processed_images']:
                try:
                    bucket_name = minio_client.buckets[bucket_type]
                    deleted_count = 0
                    
                    # Use paginator to handle large number of objects
                    paginator = minio_client.client.get_paginator('list_objects_v2')
                    pages = paginator.paginate(Bucket=bucket_name)
                    
                    objects_to_delete = []
                    for page in pages:
                        if 'Contents' in page:
                            for obj in page['Contents']:
                                objects_to_delete.append({'Key': obj['Key']})
                                
                                # Delete in batches of 1000 (AWS limit)
                                if len(objects_to_delete) >= 1000:
                                    delete_response = minio_client.client.delete_objects(
                                        Bucket=bucket_name,
                                        Delete={'Objects': objects_to_delete}
                                    )
                                    deleted_count += len(objects_to_delete)
                                    objects_to_delete = []
                    
                    # Delete remaining objects
                    if objects_to_delete:
                        delete_response = minio_client.client.delete_objects(
                            Bucket=bucket_name,
                            Delete={'Objects': objects_to_delete}
                        )
                        deleted_count += len(objects_to_delete)
                    
                    cleanup_results['minio_cleanup'][bucket_type] = f'deleted {deleted_count} objects'
                    print(f"Cleaned bucket {bucket_name}: {deleted_count} objects deleted")
                    
                except Exception as e:
                    print(f"Error cleaning bucket {bucket_type}: {str(e)}")
                    cleanup_results['minio_cleanup'][bucket_type] = f'error: {str(e)}'
                    
        except Exception as e:
            cleanup_results['errors'].append(f"MinIO cleanup error: {str(e)}")
            cleanup_results['success'] = False
        
        return jsonify(cleanup_results)
        
    except Exception as e:
        print(f"Cleanup error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

@app.route('/api/search-training-data', methods=['GET'])
def search_training_data():
    """Search training data in Qdrant"""
    try:
        category = request.args.get('category')
        limit = int(request.args.get('limit', 20))
        
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from services.qdrant_client import get_qdrant_client
        
        qdrant_client = get_qdrant_client()
        
        # Build filters
        filters = {}
        if category:
            filters['category'] = category
        
        # Count total items
        total_count = qdrant_client.count_embeddings(
            collection_type='training_embeddings',
            filters=filters
        )
        
        # Get collection info
        collection_info = qdrant_client.get_collection_info('training_embeddings')
        
        return jsonify({
            'success': True,
            'total_count': total_count,
            'collection_info': collection_info,
            'filters_applied': filters
        })
        
    except Exception as e:
        print(f"Search training data error: {str(e)}")
        return jsonify({'error': f'Failed to search training data: {str(e)}'}), 500

@app.route('/api/batch-process', methods=['POST'])
def batch_process_training_data():
    """Batch process training data from directory"""
    try:
        data = request.get_json()
        directory_path = data.get('directory_path')
        metadata_dict = data.get('metadata', {})
        confidence_threshold = data.get('confidence_threshold', 0.3)
        
        if not directory_path:
            return jsonify({'error': 'Directory path is required'}), 400
        
        if not os.path.exists(directory_path):
            return jsonify({'error': 'Directory does not exist'}), 400
        
        print(f"Starting batch processing for: {directory_path}")
        
        # Import batch processor
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from services.batch_processor import FashionBatchProcessor
        
        # Initialize processor
        processor = FashionBatchProcessor(confidence_threshold=confidence_threshold)
        
        # Find image files
        image_files = processor._find_image_files(directory_path, recursive=True)
        
        if not image_files:
            return jsonify({'error': 'No image files found in directory'}), 400
        
        processed_items = []
        total_objects = 0
        failed_count = 0
        
        # Process each image
        for i, image_path in enumerate(image_files):
            try:
                print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                # Get metadata for this image
                image_name = os.path.basename(image_path)
                image_metadata = metadata_dict.get(image_name, {})
                
                # Process single image
                result = processor.process_single_image(image_path, image_metadata)
                
                if result['success']:
                    processed_items.append({
                        'filename': result['filename'],
                        'objects_processed': result.get('objects_processed', 0),
                        'categories': result.get('categories', [])
                    })
                    total_objects += result.get('objects_processed', 0)
                else:
                    failed_count += 1
                    print(f"Failed to process {image_name}: {result.get('error')}")
                
            except Exception as e:
                failed_count += 1
                print(f"Error processing {image_path}: {str(e)}")
                continue
        
        return jsonify({
            'success': True,
            'message': f'Batch processing completed',
            'results': {
                'total_images_found': len(image_files),
                'images_processed': len(processed_items),
                'images_failed': failed_count,
                'total_objects_extracted': total_objects,
                'processed_items': processed_items
            }
        })
        
    except Exception as e:
        print(f"Batch processing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500

def allowed_file(filename):
    """Allowed file extensions"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)