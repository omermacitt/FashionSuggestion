import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import uuid
import logging
from typing import List, Dict, Optional, Union, BinaryIO
from datetime import datetime, timedelta
from io import BytesIO
import mimetypes
from pathlib import Path

logger = logging.getLogger(__name__)

class FashionMinIOClient:
    """
    MinIO S3 client for fashion image storage
    """
    
    def __init__(self, 
                 endpoint_url: str = None,
                 access_key: str = None,
                 secret_key: str = None,
                 region_name: str = 'us-east-1',
                 secure: bool = False):
        """
        Initialize MinIO client
        
        Args:
            endpoint_url: MinIO endpoint URL
            access_key: MinIO access key
            secret_key: MinIO secret key
            region_name: AWS region name
            secure: Use HTTPS connection
        """
        # Load from environment if not provided
        self.endpoint_url = endpoint_url or os.getenv('MINIO_ENDPOINT', 'http://localhost:9000')
        self.access_key = access_key or os.getenv('MINIO_ROOT_USER')
        self.secret_key = secret_key or os.getenv('MINIO_ROOT_PASSWORD')
        self.region_name = region_name
        self.secure = secure
        
        # Bucket names for different purposes
        self.buckets = {
            'user_uploads': 'user-uploads',
            'training_data': 'training-data',
            'cropped_objects': 'cropped-objects',
            'processed_images': 'processed-images'
        }
        
        self.client = None
        self._initialize_client()
        self._ensure_buckets()
    
    def _initialize_client(self):
        """Initialize S3 client with MinIO configuration"""
        try:
            self.client = boto3.client(
                's3',
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region_name,
                use_ssl=self.secure
            )
            
            # Test connection
            self.client.list_buckets()
            logger.info(f"Successfully connected to MinIO at {self.endpoint_url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {str(e)}")
            raise
    
    def _ensure_buckets(self):
        """Create buckets if they don't exist"""
        try:
            existing_buckets = [bucket['Name'] for bucket in self.client.list_buckets()['Buckets']]
            
            for bucket_purpose, bucket_name in self.buckets.items():
                if bucket_name not in existing_buckets:
                    self.client.create_bucket(Bucket=bucket_name)
                    logger.info(f"Created bucket: {bucket_name}")
                else:
                    logger.info(f"Bucket already exists: {bucket_name}")
                    
        except Exception as e:
            logger.error(f"Error ensuring buckets exist: {str(e)}")
            raise
    
    def upload_file(self, 
                   file_path: str, 
                   bucket_type: str = 'user_uploads',
                   object_name: str = None,
                   metadata: Dict = None) -> Dict:
        """
        Upload file to MinIO
        
        Args:
            file_path: Local file path
            bucket_type: Type of bucket ('user_uploads', 'training_data', etc.)
            object_name: S3 object name (auto-generated if None)
            metadata: Additional metadata
            
        Returns:
            Dictionary with upload result information
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            # Generate object name if not provided
            if object_name is None:
                file_extension = Path(file_path).suffix
                object_name = f"{uuid.uuid4().hex}{file_extension}"
            
            # Prepare metadata
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Detect content type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type:
                extra_args['ContentType'] = content_type
            
            # Upload file
            self.client.upload_file(
                file_path,
                bucket_name,
                object_name,
                ExtraArgs=extra_args
            )
            
            # Generate URLs
            file_url = f"{self.endpoint_url}/{bucket_name}/{object_name}"
            
            result = {
                'success': True,
                'bucket': bucket_name,
                'object_name': object_name,
                'file_url': file_url,
                'file_size': os.path.getsize(file_path),
                'upload_time': datetime.utcnow().isoformat()
            }
            
            logger.info(f"File uploaded successfully: {object_name} to {bucket_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise
    
    def upload_bytes(self, 
                    file_bytes: bytes, 
                    bucket_type: str = 'user_uploads',
                    object_name: str = None,
                    content_type: str = 'application/octet-stream',
                    metadata: Dict = None) -> Dict:
        """
        Upload bytes data to MinIO
        
        Args:
            file_bytes: File content as bytes
            bucket_type: Type of bucket
            object_name: S3 object name (auto-generated if None)
            content_type: MIME content type
            metadata: Additional metadata
            
        Returns:
            Dictionary with upload result information
        """
        try:
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            # Generate object name if not provided
            if object_name is None:
                extension = '.jpg' if 'image' in content_type else ''
                object_name = f"{uuid.uuid4().hex}{extension}"
            
            # Prepare metadata
            extra_args = {'ContentType': content_type}
            if metadata:
                extra_args['Metadata'] = metadata
            
            # Upload bytes
            file_obj = BytesIO(file_bytes)
            self.client.upload_fileobj(
                file_obj,
                bucket_name,
                object_name,
                ExtraArgs=extra_args
            )
            
            # Generate URLs
            file_url = f"{self.endpoint_url}/{bucket_name}/{object_name}"
            
            result = {
                'success': True,
                'bucket': bucket_name,
                'object_name': object_name,
                'file_url': file_url,
                'file_size': len(file_bytes),
                'upload_time': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Bytes uploaded successfully: {object_name} to {bucket_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error uploading bytes: {str(e)}")
            raise
    
    def download_file(self, 
                     bucket_type: str, 
                     object_name: str, 
                     local_path: str = None) -> str:
        """
        Download file from MinIO
        
        Args:
            bucket_type: Type of bucket
            object_name: S3 object name
            local_path: Local file path (auto-generated if None)
            
        Returns:
            Local file path
        """
        try:
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            # Generate local path if not provided
            if local_path is None:
                local_path = f"/tmp/{object_name}"
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            self.client.download_file(bucket_name, object_name, local_path)
            
            logger.info(f"File downloaded successfully: {object_name} from {bucket_name}")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise
    
    def download_bytes(self, bucket_type: str, object_name: str) -> bytes:
        """
        Download file as bytes from MinIO
        
        Args:
            bucket_type: Type of bucket
            object_name: S3 object name
            
        Returns:
            File content as bytes
        """
        try:
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            # Download to memory
            file_obj = BytesIO()
            self.client.download_fileobj(bucket_name, object_name, file_obj)
            
            # Get bytes
            file_bytes = file_obj.getvalue()
            
            logger.info(f"File downloaded as bytes: {object_name} from {bucket_name}")
            return file_bytes
            
        except Exception as e:
            logger.error(f"Error downloading file as bytes: {str(e)}")
            raise
    
    def delete_file(self, bucket_type: str, object_name: str) -> bool:
        """
        Delete file from MinIO
        
        Args:
            bucket_type: Type of bucket
            object_name: S3 object name
            
        Returns:
            True if successful
        """
        try:
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            self.client.delete_object(Bucket=bucket_name, Key=object_name)
            
            logger.info(f"File deleted successfully: {object_name} from {bucket_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            raise
    
    def list_files(self, 
                  bucket_type: str, 
                  prefix: str = '',
                  limit: int = 1000) -> List[Dict]:
        """
        List files in bucket
        
        Args:
            bucket_type: Type of bucket
            prefix: Object name prefix filter
            limit: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        try:
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            response = self.client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                MaxKeys=limit
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    file_info = {
                        'object_name': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'etag': obj['ETag'].strip('"'),
                        'file_url': f"{self.endpoint_url}/{bucket_name}/{obj['Key']}"
                    }
                    files.append(file_info)
            
            logger.info(f"Listed {len(files)} files from {bucket_name} with prefix '{prefix}'")
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise
    
    def get_file_info(self, bucket_type: str, object_name: str) -> Dict:
        """
        Get file information
        
        Args:
            bucket_type: Type of bucket
            object_name: S3 object name
            
        Returns:
            File information dictionary
        """
        try:
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            response = self.client.head_object(Bucket=bucket_name, Key=object_name)
            
            file_info = {
                'object_name': object_name,
                'size': response['ContentLength'],
                'content_type': response.get('ContentType', ''),
                'last_modified': response['LastModified'].isoformat(),
                'etag': response['ETag'].strip('"'),
                'metadata': response.get('Metadata', {}),
                'file_url': f"{self.endpoint_url}/{bucket_name}/{object_name}"
            }
            
            return file_info
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            raise
    
    def generate_presigned_url(self, 
                              bucket_type: str, 
                              object_name: str,
                              expiration: int = 3600,
                              method: str = 'GET') -> str:
        """
        Generate presigned URL for file access
        
        Args:
            bucket_type: Type of bucket
            object_name: S3 object name
            expiration: URL expiration time in seconds
            method: HTTP method ('GET', 'PUT')
            
        Returns:
            Presigned URL
        """
        try:
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            if method == 'GET':
                client_method = 'get_object'
            elif method == 'PUT':
                client_method = 'put_object'
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            url = self.client.generate_presigned_url(
                client_method,
                Params={'Bucket': bucket_name, 'Key': object_name},
                ExpiresIn=expiration
            )
            
            logger.info(f"Generated presigned URL for {object_name} (expires in {expiration}s)")
            return url
            
        except Exception as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            raise
    
    def copy_file(self, 
                 source_bucket_type: str, 
                 source_object: str,
                 dest_bucket_type: str, 
                 dest_object: str = None) -> Dict:
        """
        Copy file between buckets
        
        Args:
            source_bucket_type: Source bucket type
            source_object: Source object name
            dest_bucket_type: Destination bucket type
            dest_object: Destination object name (same as source if None)
            
        Returns:
            Copy result information
        """
        try:
            source_bucket = self.buckets.get(source_bucket_type)
            dest_bucket = self.buckets.get(dest_bucket_type)
            
            if not source_bucket or not dest_bucket:
                raise ValueError("Invalid bucket type")
            
            if dest_object is None:
                dest_object = source_object
            
            # Copy object
            copy_source = {'Bucket': source_bucket, 'Key': source_object}
            self.client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_object
            )
            
            result = {
                'success': True,
                'source_bucket': source_bucket,
                'source_object': source_object,
                'dest_bucket': dest_bucket,
                'dest_object': dest_object,
                'file_url': f"{self.endpoint_url}/{dest_bucket}/{dest_object}",
                'copy_time': datetime.utcnow().isoformat()
            }
            
            logger.info(f"File copied: {source_object} from {source_bucket} to {dest_bucket}")
            return result
            
        except Exception as e:
            logger.error(f"Error copying file: {str(e)}")
            raise
    
    def get_bucket_info(self, bucket_type: str) -> Dict:
        """
        Get bucket information
        
        Args:
            bucket_type: Type of bucket
            
        Returns:
            Bucket information dictionary
        """
        try:
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            # Get bucket location
            location = self.client.get_bucket_location(Bucket=bucket_name)
            
            # Count objects and calculate total size
            response = self.client.list_objects_v2(Bucket=bucket_name)
            object_count = response.get('KeyCount', 0)
            total_size = sum(obj['Size'] for obj in response.get('Contents', []))
            
            bucket_info = {
                'bucket_name': bucket_name,
                'bucket_type': bucket_type,
                'location': location.get('LocationConstraint', 'us-east-1'),
                'object_count': object_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            }
            
            return bucket_info
            
        except Exception as e:
            logger.error(f"Error getting bucket info: {str(e)}")
            raise
    
    def cleanup_old_files(self, 
                         bucket_type: str, 
                         days_old: int = 30,
                         prefix: str = '') -> int:
        """
        Clean up old files from bucket
        
        Args:
            bucket_type: Type of bucket
            days_old: Delete files older than this many days
            prefix: Object name prefix filter
            
        Returns:
            Number of files deleted
        """
        try:
            bucket_name = self.buckets.get(bucket_type)
            if not bucket_name:
                raise ValueError(f"Invalid bucket type: {bucket_type}")
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            response = self.client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            deleted_count = 0
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                        self.client.delete_object(Bucket=bucket_name, Key=obj['Key'])
                        deleted_count += 1
                        logger.info(f"Deleted old file: {obj['Key']}")
            
            logger.info(f"Cleanup completed: {deleted_count} files deleted from {bucket_name}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise


# Singleton instance for global use
_minio_client_instance = None

def get_minio_client(endpoint_url: str = None,
                    access_key: str = None,
                    secret_key: str = None,
                    region_name: str = 'us-east-1',
                    secure: bool = False) -> FashionMinIOClient:
    """
    Get singleton MinIO client instance
    
    Args:
        endpoint_url: MinIO endpoint URL
        access_key: MinIO access key
        secret_key: MinIO secret key
        region_name: AWS region name
        secure: Use HTTPS connection
        
    Returns:
        FashionMinIOClient instance
    """
    global _minio_client_instance
    
    if _minio_client_instance is None:
        _minio_client_instance = FashionMinIOClient(
            endpoint_url, access_key, secret_key, region_name, secure
        )
    
    return _minio_client_instance