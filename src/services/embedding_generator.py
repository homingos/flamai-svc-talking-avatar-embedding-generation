import torch
import logging
from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Optimized embedding generation class with GPU support and batch processing.
    Focuses solely on embedding creation from text inputs.
    """
    
    def __init__(
        self, 
        model_name: str = 'all-MiniLM-L6-v2',
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: int = 512,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
            batch_size: Batch size for processing multiple texts
            max_seq_length: Maximum sequence length for tokenization
            normalize_embeddings: Whether to normalize embeddings to unit vectors
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.normalize_embeddings = normalize_embeddings
        
        # Auto-detect device if not specified or set to "auto"
        if device is None or device == "auto":
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Batch size: {batch_size}")
        
        # Initialize the model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the SentenceTransformer model with optimizations."""
        try:
            logger.info("Loading SentenceTransformer model...")
            
            # Load model with optimizations
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Set model to evaluation mode for inference
            self.model.eval()
            
            # Optimize for inference
            if self.device == 'cuda':
                # Enable mixed precision for faster inference on GPU
                self.model.half()
                logger.info("Enabled half precision for GPU optimization")
            
            # Set max sequence length
            self.model.max_seq_length = self.max_seq_length
            
            logger.info("✅ Embedding model loaded and optimized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text string
            
        Returns:
            numpy array containing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
            
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of text strings with optimized processing.
        
        Args:
            texts: List of input text strings
            batch_size: Override default batch size for this operation
            
        Returns:
            List of numpy arrays containing embedding vectors
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts found in input list")
            
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        # Use provided batch size or default
        effective_batch_size = batch_size or self.batch_size
        
        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts with batch size {effective_batch_size}")
            
            # Generate embeddings with batch processing
            embeddings = self.model.encode(
                valid_texts,
                batch_size=effective_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True,
                device=self.device
            )
            
            # Convert to list of numpy arrays if needed
            if isinstance(embeddings, np.ndarray):
                embeddings = [embeddings[i] for i in range(len(embeddings))]
            
            logger.info(f"✅ Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to generate batch embeddings: {str(e)}")
            raise
    
    def generate_embeddings_parallel(
        self, 
        texts: List[str], 
        num_workers: int = 4,
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings using parallel processing for large datasets.
        
        Args:
            texts: List of input text strings
            num_workers: Number of parallel workers
            batch_size: Batch size for each worker
            
        Returns:
            List of numpy arrays containing embedding vectors
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
            
        if len(texts) < num_workers * 2:
            # For small datasets, use regular batch processing
            return self.generate_embeddings_batch(texts, batch_size)
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts using {num_workers} parallel workers")
            
            # Split texts into chunks for parallel processing
            chunk_size = len(texts) // num_workers
            text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
            
            # Process chunks in parallel
            all_embeddings = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit tasks
                future_to_chunk = {
                    executor.submit(self.generate_embeddings_batch, chunk, batch_size): chunk 
                    for chunk in text_chunks
                }
                
                # Collect results
                for future in future_to_chunk:
                    try:
                        chunk_embeddings = future.result()
                        all_embeddings.extend(chunk_embeddings)
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        raise
            
            logger.info(f"✅ Successfully generated {len(all_embeddings)} embeddings using parallel processing")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to generate parallel embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model.
        
        Returns:
            Integer representing the embedding dimension
        """
        try:
            # Generate a test embedding to get dimension
            test_embedding = self.generate_embedding("test")
            return len(test_embedding)
        except Exception as e:
            logger.error(f"❌ Failed to get embedding dimension: {str(e)}")
            raise
    
    def optimize_for_inference(self):
        """Apply additional optimizations for inference speed."""
        try:
            if self.device == 'cuda':
                # Enable cuDNN benchmark for consistent input sizes
                torch.backends.cudnn.benchmark = True
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                logger.info("✅ Applied GPU inference optimizations")
            else:
                logger.info("✅ Model optimized for CPU inference")
                
        except Exception as e:
            logger.error(f"❌ Failed to apply optimizations: {str(e)}")
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        try:
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("✅ Cache cleared successfully")
        except Exception as e:
            logger.error(f"❌ Failed to clear cache: {str(e)}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "normalize_embeddings": self.normalize_embeddings,
            "embedding_dimension": self.get_embedding_dimension(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the embedding generator
    generator = EmbeddingGenerator(
        model_name='all-MiniLM-L6-v2',
        batch_size=16,
        device='cuda'  # or 'cpu' or None for auto-detection
    )
    
    # Apply optimizations
    generator.optimize_for_inference()
    
    # Example 1: Single text embedding
    text = "This is a sample text for embedding generation"
    embedding = generator.generate_embedding(text)
    print(f"Single embedding shape: {embedding.shape}")
    
    # Example 2: Batch processing
    texts = [
        "First sample text",
        "Second sample text", 
        "Third sample text",
        "Fourth sample text"
    ]
    embeddings = generator.generate_embeddings_batch(texts)
    print(f"Batch embeddings count: {len(embeddings)}")
    print(f"Each embedding shape: {embeddings[0].shape}")
    
    # Example 3: Large dataset with parallel processing
    large_texts = [f"Sample text number {i}" for i in range(100)]
    parallel_embeddings = generator.generate_embeddings_parallel(large_texts, num_workers=4)
    print(f"Parallel embeddings count: {len(parallel_embeddings)}")
    
    # Get model information
    model_info = generator.get_model_info()
    print(f"Model info: {model_info}")
    
    # Clear cache when done
    generator.clear_cache()
