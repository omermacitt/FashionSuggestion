import os
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, SearchRequest
)
import logging
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class FashionQdrantClient:
    """
    Qdrant vector database client for fashion embeddings
    """
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 grpc_port: int = None,
                 prefer_grpc: bool = False,
                 https: bool = False,
                 api_key: str = None):
        """
        Initialize Qdrant client
        
        Args:
            host: Qdrant host
            port: Qdrant HTTP port
            grpc_port: Qdrant gRPC port
            prefer_grpc: Use gRPC instead of HTTP
            https: Use HTTPS connection
            api_key: API key for authentication
        """
        # Load from environment if not provided
        self.host = host or os.getenv('QDRANT_HOST', 'localhost')
        self.port = port or int(os.getenv('QDRANT_PORT', '6333'))
        self.grpc_port = grpc_port or int(os.getenv('QDRANT_GRPC_PORT', '6334'))
        self.prefer_grpc = prefer_grpc
        self.https = https
        self.api_key = api_key
        
        # Collection names for different purposes
        self.collections = {
            'fashion_embeddings': 'fashion-embeddings',
            'training_embeddings': 'training-embeddings',
            'user_embeddings': 'user-embeddings'
        }
        
        self.client = None
        self.embedding_dim = 512  # CLIP ViT-B/32 dimension
        self._initialize_client()
        self._ensure_collections()
    
    def _initialize_client(self):
        """Initialize Qdrant client"""
        try:
            if self.prefer_grpc:
                self.client = QdrantClient(
                    host=self.host,
                    grpc_port=self.grpc_port,
                    prefer_grpc=True,
                    https=self.https,
                    api_key=self.api_key
                )
            else:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    https=self.https,
                    api_key=self.api_key
                )
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Successfully connected to Qdrant at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise
    
    def _ensure_collections(self):
        """Create collections if they don't exist"""
        try:
            existing_collections = [col.name for col in self.client.get_collections().collections]
            
            for collection_purpose, collection_name in self.collections.items():
                if collection_name not in existing_collections:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=self.embedding_dim,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created collection: {collection_name}")
                else:
                    logger.info(f"Collection already exists: {collection_name}")
                    
        except Exception as e:
            logger.error(f"Error ensuring collections exist: {str(e)}")
            raise
    
    def store_embedding(self, 
                       embedding: np.ndarray,
                       metadata: Dict,
                       collection_type: str = 'fashion_embeddings',
                       point_id: str = None) -> str:
        """
        Store a single embedding with metadata
        
        Args:
            embedding: CLIP embedding vector (512-dim)
            metadata: Metadata dictionary
            collection_type: Collection type ('fashion_embeddings', etc.)
            point_id: Custom point ID (auto-generated if None)
            
        Returns:
            Point ID
        """
        try:
            collection_name = self.collections.get(collection_type)
            if not collection_name:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
            # Generate point ID if not provided
            if point_id is None:
                point_id = str(uuid.uuid4())
            
            # Ensure embedding is the right format
            if isinstance(embedding, np.ndarray):
                embedding = embedding.flatten().tolist()
            
            if len(embedding) != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding)}")
            
            # Add timestamp to metadata
            metadata['created_at'] = datetime.utcnow().isoformat()
            metadata['point_id'] = point_id
            
            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload=metadata
            )
            
            # Upsert point
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            
            logger.info(f"Stored embedding with ID: {point_id} in {collection_name}")
            return point_id
            
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            raise
    
    def store_embeddings_batch(self, 
                              embeddings: List[np.ndarray],
                              metadata_list: List[Dict],
                              collection_type: str = 'fashion_embeddings',
                              point_ids: List[str] = None) -> List[str]:
        """
        Store multiple embeddings in batch
        
        Args:
            embeddings: List of CLIP embedding vectors
            metadata_list: List of metadata dictionaries
            collection_type: Collection type
            point_ids: List of custom point IDs (auto-generated if None)
            
        Returns:
            List of point IDs
        """
        try:
            collection_name = self.collections.get(collection_type)
            if not collection_name:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
            if len(embeddings) != len(metadata_list):
                raise ValueError("Number of embeddings and metadata must match")
            
            # Generate point IDs if not provided
            if point_ids is None:
                point_ids = [str(uuid.uuid4()) for _ in embeddings]
            elif len(point_ids) != len(embeddings):
                raise ValueError("Number of point IDs must match embeddings")
            
            points = []
            current_time = datetime.utcnow().isoformat()
            
            for i, (embedding, metadata) in enumerate(zip(embeddings, metadata_list)):
                # Ensure embedding is the right format
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.flatten().tolist()
                
                if len(embedding) != self.embedding_dim:
                    raise ValueError(f"Embedding {i} dimension mismatch")
                
                # Add timestamp and ID to metadata
                metadata['created_at'] = current_time
                metadata['point_id'] = point_ids[i]
                
                point = PointStruct(
                    id=point_ids[i],
                    vector=embedding,
                    payload=metadata
                )
                points.append(point)
            
            # Batch upsert
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Stored {len(points)} embeddings in batch to {collection_name}")
            return point_ids
            
        except Exception as e:
            logger.error(f"Error storing embeddings batch: {str(e)}")
            raise
    
    def search_similar(self, 
                      query_embedding: np.ndarray,
                      collection_type: str = 'fashion_embeddings',
                      limit: int = 10,
                      score_threshold: float = 0.7,
                      filters: Dict = None) -> List[Dict]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            collection_type: Collection to search in
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filters: Additional filters (e.g., {'category': 'dress'})
            
        Returns:
            List of similar items with scores and metadata
        """
        try:
            collection_name = self.collections.get(collection_type)
            if not collection_name:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
            # Ensure query embedding is the right format
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.flatten().tolist()
            
            if len(query_embedding) != self.embedding_dim:
                raise ValueError(f"Query embedding dimension mismatch")
            
            # Build filter conditions
            filter_conditions = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                if conditions:
                    filter_conditions = Filter(must=conditions)
            
            # Search
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=filter_conditions,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for scored_point in search_result:
                result = {
                    'id': scored_point.id,
                    'score': scored_point.score,
                    'metadata': scored_point.payload
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar items in {collection_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {str(e)}")
            raise
    
    def search_by_categories(self, 
                           query_embedding: np.ndarray,
                           categories: List[str],
                           collection_type: str = 'fashion_embeddings',
                           limit_per_category: int = 3) -> Dict[str, List[Dict]]:
        """
        Search for similar items in specific categories
        
        Args:
            query_embedding: Query embedding vector
            categories: List of categories to search in
            collection_type: Collection to search in
            limit_per_category: Max results per category
            
        Returns:
            Dictionary with category as key and similar items as value
        """
        try:
            results = {}
            
            for category in categories:
                filters = {'category': category}
                category_results = self.search_similar(
                    query_embedding=query_embedding,
                    collection_type=collection_type,
                    limit=limit_per_category,
                    filters=filters
                )
                results[category] = category_results
            
            logger.info(f"Searched in {len(categories)} categories")
            return results
            
        except Exception as e:
            logger.error(f"Error searching by categories: {str(e)}")
            raise
    
    def get_embedding(self, 
                     point_id: str,
                     collection_type: str = 'fashion_embeddings') -> Dict:
        """
        Get embedding and metadata by ID
        
        Args:
            point_id: Point ID
            collection_type: Collection type
            
        Returns:
            Dictionary with embedding and metadata
        """
        try:
            collection_name = self.collections.get(collection_type)
            if not collection_name:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_vectors=True
            )
            
            if not result:
                raise ValueError(f"Point not found: {point_id}")
            
            point = result[0]
            return {
                'id': point.id,
                'vector': point.vector,
                'metadata': point.payload
            }
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise
    
    def delete_embedding(self, 
                        point_id: str,
                        collection_type: str = 'fashion_embeddings') -> bool:
        """
        Delete embedding by ID
        
        Args:
            point_id: Point ID to delete
            collection_type: Collection type
            
        Returns:
            True if successful
        """
        try:
            collection_name = self.collections.get(collection_type)
            if not collection_name:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
            
            logger.info(f"Deleted embedding: {point_id} from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embedding: {str(e)}")
            raise
    
    def update_metadata(self, 
                       point_id: str,
                       metadata: Dict,
                       collection_type: str = 'fashion_embeddings') -> bool:
        """
        Update metadata for existing point
        
        Args:
            point_id: Point ID
            metadata: New metadata
            collection_type: Collection type
            
        Returns:
            True if successful
        """
        try:
            collection_name = self.collections.get(collection_type)
            if not collection_name:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
            # Add update timestamp
            metadata['updated_at'] = datetime.utcnow().isoformat()
            
            self.client.set_payload(
                collection_name=collection_name,
                payload=metadata,
                points=[point_id]
            )
            
            logger.info(f"Updated metadata for: {point_id} in {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            raise
    
    def count_embeddings(self, 
                        collection_type: str = 'fashion_embeddings',
                        filters: Dict = None) -> int:
        """
        Count embeddings in collection
        
        Args:
            collection_type: Collection type
            filters: Optional filters
            
        Returns:
            Number of embeddings
        """
        try:
            collection_name = self.collections.get(collection_type)
            if not collection_name:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
            # Build filter conditions
            filter_conditions = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                if conditions:
                    filter_conditions = Filter(must=conditions)
            
            result = self.client.count(
                collection_name=collection_name,
                count_filter=filter_conditions
            )
            
            return result.count
            
        except Exception as e:
            logger.error(f"Error counting embeddings: {str(e)}")
            raise
    
    def get_collection_info(self, collection_type: str = 'fashion_embeddings') -> Dict:
        """
        Get collection information
        
        Args:
            collection_type: Collection type
            
        Returns:
            Collection information dictionary
        """
        try:
            collection_name = self.collections.get(collection_type)
            if not collection_name:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
            info = self.client.get_collection(collection_name)
            
            return {
                'name': info.config.params.vectors.size,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'points_count': info.points_count,
                'segments_count': info.segments_count,
                'config': {
                    'distance': info.config.params.vectors.distance.value,
                    'vector_size': info.config.params.vectors.size
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise
    
    def create_index(self, 
                    collection_type: str = 'fashion_embeddings',
                    field_name: str = 'category',
                    field_type: str = 'keyword') -> bool:
        """
        Create index on metadata field for faster filtering
        
        Args:
            collection_type: Collection type
            field_name: Field name to index
            field_type: Field type ('keyword', 'integer', 'float', 'bool')
            
        Returns:
            True if successful
        """
        try:
            collection_name = self.collections.get(collection_type)
            if not collection_name:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
            # Create index based on field type
            if field_type == 'keyword':
                index_type = models.KeywordIndexParams()
            elif field_type == 'integer':
                index_type = models.IntegerIndexParams()
            elif field_type == 'float':
                index_type = models.FloatIndexParams()
            elif field_type == 'bool':
                index_type = models.BoolIndexParams()
            else:
                raise ValueError(f"Unsupported field type: {field_type}")
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=index_type
            )
            
            logger.info(f"Created index on {field_name} in {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
    
    def cleanup_old_embeddings(self, 
                              collection_type: str = 'fashion_embeddings',
                              days_old: int = 30) -> int:
        """
        Clean up old embeddings
        
        Args:
            collection_type: Collection type
            days_old: Delete embeddings older than this many days
            
        Returns:
            Number of embeddings deleted
        """
        try:
            from datetime import timedelta
            cutoff_date = (datetime.utcnow() - timedelta(days=days_old)).isoformat()
            
            collection_name = self.collections.get(collection_type)
            if not collection_name:
                raise ValueError(f"Invalid collection type: {collection_type}")
            
            # Search for old embeddings
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="created_at",
                        range=models.Range(lt=cutoff_date)
                    )
                ]
            )
            
            # Get old points
            old_points = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=filter_condition,
                limit=1000,
                with_payload=False,
                with_vectors=False
            )[0]
            
            if old_points:
                # Delete old points
                old_ids = [point.id for point in old_points]
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(points=old_ids)
                )
                
                logger.info(f"Deleted {len(old_ids)} old embeddings from {collection_name}")
                return len(old_ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error cleaning up old embeddings: {str(e)}")
            raise


# Singleton instance for global use
_qdrant_client_instance = None

def get_qdrant_client(host: str = None,
                     port: int = None,
                     grpc_port: int = None,
                     prefer_grpc: bool = False,
                     https: bool = False,
                     api_key: str = None) -> FashionQdrantClient:
    """
    Get singleton Qdrant client instance
    
    Args:
        host: Qdrant host
        port: Qdrant HTTP port
        grpc_port: Qdrant gRPC port
        prefer_grpc: Use gRPC instead of HTTP
        https: Use HTTPS connection
        api_key: API key for authentication
        
    Returns:
        FashionQdrantClient instance
    """
    global _qdrant_client_instance
    
    if _qdrant_client_instance is None:
        _qdrant_client_instance = FashionQdrantClient(
            host, port, grpc_port, prefer_grpc, https, api_key
        )
    
    return _qdrant_client_instance


# Fashion-specific metadata schemas
class FashionMetadataSchema:
    """
    Standard metadata schemas for fashion items
    """
    
    @staticmethod
    def product_metadata(product_id: str,
                        category: str,
                        subcategory: str = None,
                        brand: str = None,
                        color: str = None,
                        material: str = None,
                        size: str = None,
                        price: float = None,
                        image_url: str = None,
                        description: str = None) -> Dict:
        """
        Create product metadata
        
        Args:
            product_id: Unique product identifier
            category: Main category (e.g., 'dress', 'shirt', 'shoes')
            subcategory: Subcategory (e.g., 'evening_dress', 'casual_shirt')
            brand: Brand name
            color: Primary color
            material: Material description
            size: Size information
            price: Product price
            image_url: URL to product image
            description: Product description
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'product_id': product_id,
            'category': category,
            'type': 'product'
        }
        
        # Add optional fields if provided
        optional_fields = {
            'subcategory': subcategory,
            'brand': brand,
            'color': color,
            'material': material,
            'size': size,
            'price': price,
            'image_url': image_url,
            'description': description
        }
        
        for key, value in optional_fields.items():
            if value is not None:
                metadata[key] = value
        
        return metadata
    
    @staticmethod
    def user_upload_metadata(user_id: str,
                           upload_id: str,
                           detected_category: str,
                           confidence: float,
                           bbox: Dict = None,
                           original_image_url: str = None,
                           cropped_image_url: str = None) -> Dict:
        """
        Create user upload metadata
        
        Args:
            user_id: User identifier
            upload_id: Upload session identifier
            detected_category: YOLO detected category
            confidence: Detection confidence score
            bbox: Bounding box coordinates
            original_image_url: URL to original image
            cropped_image_url: URL to cropped image
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            'user_id': user_id,
            'upload_id': upload_id,
            'category': detected_category,
            'confidence': confidence,
            'type': 'user_upload'
        }
        
        # Add optional fields if provided
        if bbox:
            metadata['bbox'] = bbox
        if original_image_url:
            metadata['original_image_url'] = original_image_url
        if cropped_image_url:
            metadata['cropped_image_url'] = cropped_image_url
        
        return metadata