#!/usr/bin/env python3
"""
Embedding Generator Script

This script generates embeddings for text data and outputs them in JSON format.
It processes input data and creates embeddings using the EmbeddingGenerator class.
"""

import json
import sys
import os
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add the src directory to the Python path to import our services
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.embedding_generator import EmbeddingGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    Main class for processing text data and generating embeddings in JSON format.
    """
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = 'auto',
        batch_size: int = 32,
        normalize_embeddings: bool = True
    ):
        """
        Initialize the embedding processor.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
            device: Device to run on ('cuda', 'cpu', or 'auto')
            batch_size: Batch size for processing multiple texts
            normalize_embeddings: Whether to normalize embeddings to unit vectors
        """
        self.generator = EmbeddingGenerator(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings
        )
        
        # Apply optimizations
        self.generator.optimize_for_inference()
        
        logger.info("✅ EmbeddingProcessor initialized successfully")
    
    def process_texts_from_file(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Process texts from a JSON file and generate embeddings.
        
        Args:
            input_file: Path to the input JSON file
            
        Returns:
            List of dictionaries containing text, vector, and id
        """
        try:
            # Load input data
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} texts from {input_file}")
            
            # Extract texts and IDs
            texts = []
            ids = []
            
            for item in data:
                if 'text' in item and 'id' in item:
                    texts.append(item['text'])
                    ids.append(item['id'])
                else:
                    logger.warning(f"Skipping item without 'text' or 'id': {item}")
            
            if not texts:
                raise ValueError("No valid texts found in input file")
            
            logger.info(f"Processing {len(texts)} texts for embedding generation")
            
            # Generate embeddings in batches
            embeddings = self.generator.generate_embeddings_batch(texts)
            
            # Create output format
            results = []
            for i, (text, embedding, text_id) in enumerate(zip(texts, embeddings, ids)):
                result = {
                    "text": text,
                    "vector": embedding.tolist(),  # Convert numpy array to list
                    "id": text_id
                }
                results.append(result)
            
            logger.info(f"✅ Successfully generated {len(results)} embeddings")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to process texts from file: {str(e)}")
            raise
    
    def process_texts_from_list(self, texts: List[str], start_id: int = 1) -> List[Dict[str, Any]]:
        """
        Process a list of texts and generate embeddings.
        
        Args:
            texts: List of text strings
            start_id: Starting ID for the generated embeddings
            
        Returns:
            List of dictionaries containing text, vector, and id
        """
        try:
            if not texts:
                raise ValueError("No texts provided")
            
            logger.info(f"Processing {len(texts)} texts for embedding generation")
            
            # Generate embeddings in batches
            embeddings = self.generator.generate_embeddings_batch(texts)
            
            # Create output format
            results = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                result = {
                    "text": text,
                    "vector": embedding.tolist(),  # Convert numpy array to list
                    "id": start_id + i
                }
                results.append(result)
            
            logger.info(f"✅ Successfully generated {len(results)} embeddings")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to process texts from list: {str(e)}")
            raise
    
    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """
        Save results to a JSON file.
        
        Args:
            results: List of embedding results
            output_file: Path to the output JSON file
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return self.generator.get_model_info()


def main():
    """Main function to run the embedding generator."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for text data and output in JSON format"
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input JSON file containing texts with IDs'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSON file for embeddings'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='all-MiniLM-L6-v2',
        help='SentenceTransformer model name (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for processing (default: auto)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Disable embedding normalization'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize processor
        logger.info("Initializing embedding processor...")
        processor = EmbeddingProcessor(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            normalize_embeddings=not args.no_normalize
        )
        
        # Print model info
        model_info = processor.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        # Process texts
        logger.info(f"Processing texts from {args.input}")
        results = processor.process_texts_from_file(args.input)
        
        # Save results
        logger.info(f"Saving results to {args.output}")
        processor.save_results(results, args.output)
        
        # Print summary
        logger.info("=" * 50)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Total embeddings generated: {len(results)}")
        logger.info(f"Embedding dimension: {len(results[0]['vector'])}")
        logger.info(f"Model used: {args.model}")
        logger.info(f"Device used: {model_info['device']}")
        
        # Clear cache
        processor.generator.clear_cache()
        
    except Exception as e:
        logger.error(f"❌ Script failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
