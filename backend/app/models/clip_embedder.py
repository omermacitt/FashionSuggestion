import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import logging
from typing import List, Union, Optional, Tuple
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class FashionCLIPEmbedder:
    """
    Fashion embedding generator using CLIP ViT-B/32 model
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        Initialize CLIP embedder
        
        Args:
            model_name: CLIP model name from Hugging Face
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.embedding_dim = 512  # ViT-B/32 embedding dimension
        
        self._load_model()
    
    def _load_model(self):
        """Load CLIP model and processor"""
        try:
            logger.info(f"Loading CLIP model: {self.model_name} on {self.device}")
            
            # Load model and processor
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"CLIP model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            raise
    
    def encode_image(self, image: Union[str, np.ndarray, Image.Image, bytes]) -> np.ndarray:
        """
        Generate embedding for a single image
        
        Args:
            image: Image path, numpy array, PIL Image, or bytes
            
        Returns:
            Normalized embedding vector as numpy array
        """
        try:
            # Convert input to PIL Image
            pil_image = self._convert_to_pil(image)
            
            # Process image
            inputs = self.processor(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize the embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embedding = image_features.cpu().numpy().flatten()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise
    
    def encode_images_batch(self, images: List[Union[str, np.ndarray, Image.Image, bytes]], 
                           batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple images in batches
        
        Args:
            images: List of images (paths, arrays, PIL Images, or bytes)
            batch_size: Number of images to process at once
            
        Returns:
            Array of embeddings with shape (num_images, embedding_dim)
        """
        try:
            all_embeddings = []
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                
                # Convert batch to PIL Images
                pil_images = [self._convert_to_pil(img) for img in batch]
                
                # Process batch
                inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    
                    # Normalize the embeddings
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Convert to numpy
                    batch_embeddings = image_features.cpu().numpy()
                    all_embeddings.append(batch_embeddings)
            
            # Concatenate all batches
            embeddings = np.vstack(all_embeddings)
            
            logger.info(f"Generated embeddings for {len(images)} images")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding image batch: {str(e)}")
            raise
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embedding for text (useful for text-to-image search)
        
        Args:
            text: Text string or list of text strings
            
        Returns:
            Normalized embedding vector(s) as numpy array
        """
        try:
            # Ensure text is a list
            if isinstance(text, str):
                text = [text]
            
            # Process text
            inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                
                # Normalize the embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embeddings = text_features.cpu().numpy()
            
            # Return flattened array if single text, otherwise return all embeddings
            if len(text) == 1:
                return embeddings.flatten()
            else:
                return embeddings
                
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2)
            
            # Convert to 0-1 range
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: np.ndarray, 
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples sorted by similarity
        """
        try:
            # Normalize query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Normalize candidate embeddings
            candidate_embeddings = candidate_embeddings / np.linalg.norm(
                candidate_embeddings, axis=1, keepdims=True
            )
            
            # Compute similarities
            similarities = np.dot(candidate_embeddings, query_embedding)
            
            # Convert to 0-1 range
            similarities = (similarities + 1) / 2
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Return results as (index, similarity) tuples
            results = [(int(idx), float(similarities[idx])) for idx in top_indices]
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding similar embeddings: {str(e)}")
            raise
    
    def _convert_to_pil(self, image: Union[str, np.ndarray, Image.Image, bytes]) -> Image.Image:
        """Convert various image formats to PIL Image"""
        try:
            if isinstance(image, str):
                # File path
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                return Image.open(image).convert('RGB')
            
            elif isinstance(image, np.ndarray):
                # OpenCV/numpy array
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR to RGB conversion for OpenCV
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(image).convert('RGB')
            
            elif isinstance(image, Image.Image):
                # PIL Image
                return image.convert('RGB')
            
            elif isinstance(image, bytes):
                # Bytes data
                return Image.open(BytesIO(image)).convert('RGB')
            
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
                
        except Exception as e:
            logger.error(f"Error converting image to PIL: {str(e)}")
            raise
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """Save embeddings to file"""
        try:
            np.save(filepath, embeddings)
            logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """Load embeddings from file"""
        try:
            embeddings = np.load(filepath)
            logger.info(f"Embeddings loaded from {filepath}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
            raise
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
    
    def get_device(self) -> str:
        """Get current device"""
        return self.device


# Singleton instance for global use
_embedder_instance = None

def get_embedder(model_name: str = "openai/clip-vit-base-patch32", 
                device: str = None) -> FashionCLIPEmbedder:
    """
    Get singleton CLIP embedder instance
    
    Args:
        model_name: CLIP model name
        device: Device to run model on
        
    Returns:
        FashionCLIPEmbedder instance
    """
    global _embedder_instance
    
    if _embedder_instance is None:
        _embedder_instance = FashionCLIPEmbedder(model_name, device)
    
    return _embedder_instance


class FashionTextQueries:
    """
    Predefined text queries for fashion items
    """
    
    FASHION_CATEGORIES = {
        'tops': [
            "a stylish shirt",
            "a fashionable blouse", 
            "a trendy t-shirt",
            "a elegant sweater",
            "a casual hoodie"
        ],
        'bottoms': [
            "stylish pants",
            "fashionable jeans",
            "elegant trousers",
            "trendy shorts",
            "stylish skirt"
        ],
        'dresses': [
            "a beautiful dress",
            "an elegant gown",
            "a stylish sundress",
            "a trendy mini dress",
            "a formal evening dress"
        ],
        'shoes': [
            "stylish shoes",
            "elegant heels",
            "casual sneakers",
            "fashionable boots",
            "trendy sandals"
        ],
        'accessories': [
            "a stylish bag",
            "elegant jewelry",
            "a fashionable hat",
            "trendy sunglasses",
            "a stylish watch"
        ]
    }
    
    @classmethod
    def get_category_queries(cls, category: str) -> List[str]:
        """Get text queries for a specific category"""
        return cls.FASHION_CATEGORIES.get(category, [])
    
    @classmethod
    def get_all_queries(cls) -> List[str]:
        """Get all fashion text queries"""
        all_queries = []
        for queries in cls.FASHION_CATEGORIES.values():
            all_queries.extend(queries)
        return all_queries