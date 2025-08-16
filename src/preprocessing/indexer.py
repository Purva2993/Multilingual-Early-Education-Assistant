"""
Preprocessing and indexing module for educational content.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langdetect import detect
from loguru import logger

from ..config import settings
from ..data_crawler.scraper import ScrapedContent, DataManager


class TextPreprocessor:
    """Text preprocessing utilities."""
    
    def __init__(self):
        self.min_chunk_size = 50
        self.max_chunk_size = settings.CHUNK_SIZE
        self.overlap = settings.CHUNK_OVERLAP
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\'""]', ' ', text)
        
        # Remove multiple consecutive punctuation
        text = re.sub(r'[.,!?;:]{2,}', '.', text)
        
        # Strip and normalize
        text = text.strip()
        
        return text
    
    def detect_language(self, text: str) -> str:
        """Detect text language."""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English
    
    def chunk_text(self, text: str, title: str = "") -> List[Dict[str, Any]]:
        """Split text into chunks with overlap."""
        if not text:
            return []
        
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Add title to first chunk if available
        if title:
            title_tokens = len(title.split())
            if title_tokens < self.max_chunk_size // 4:
                current_chunk.append(title)
                current_length += title_tokens
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed max chunk size, finalize current chunk
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                if current_length >= self.min_chunk_size:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'length': current_length,
                        'sentences': len(current_chunk)
                    })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the final chunk
        if current_chunk and current_length >= self.min_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'length': current_length,
                'sentences': len(current_chunk)
            })
        
        return chunks


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self.model:
            return self.model.get_sentence_embedding_dimension()
        return 384  # Default dimension for MiniLM


class VectorDatabase:
    """Vector database for storing and searching embeddings."""
    
    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or settings.VECTOR_DB_PATH)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.metadata = []
        self.embedding_generator = EmbeddingGenerator()
        self.dimension = self.embedding_generator.get_embedding_dimension()
        
        # File paths
        self.index_file = self.db_path / "faiss_index.idx"
        self.metadata_file = self.db_path / "metadata.pkl"
        self.config_file = self.db_path / "config.json"
        
        self._load_or_create_index()
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        if self.index_file.exists() and self.metadata_file.exists():
            self._load_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        logger.info(f"Creating new FAISS index with dimension {self.dimension}")
        
        # Use IndexFlatIP for cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        
        # Save config
        config = {
            'dimension': self.dimension,
            'index_type': 'IndexFlatIP',
            'created_at': datetime.now().isoformat(),
            'embedding_model': self.embedding_generator.model_name
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _load_index(self):
        """Load existing FAISS index."""
        try:
            logger.info("Loading existing FAISS index")
            self.index = faiss.read_index(str(self.index_file))
            
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            self._create_new_index()
    
    def _save_index(self):
        """Save the FAISS index and metadata."""
        try:
            faiss.write_index(self.index, str(self.index_file))
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info(f"Saved index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector database."""
        if not documents:
            return
        
        logger.info(f"Adding {len(documents)} documents to vector database")
        
        # Extract texts for embedding
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata
        for i, doc in enumerate(documents):
            metadata = {
                'id': len(self.metadata),
                'text': doc['text'],
                'source_url': doc.get('source_url', ''),
                'source_name': doc.get('source_name', ''),
                'title': doc.get('title', ''),
                'language': doc.get('language', 'en'),
                'chunk_index': doc.get('chunk_index', 0),
                'added_at': datetime.now().isoformat()
            }
            self.metadata.append(metadata)
        
        # Save the updated index
        self._save_index()
        
        logger.info(f"Successfully added {len(documents)} documents")
    
    def search(self, query: str, k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embeddings([query])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= threshold:
                metadata = self.metadata[idx].copy()
                metadata['similarity_score'] = float(score)
                results.append(metadata)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'total_metadata': len(self.metadata),
            'embedding_model': self.embedding_generator.model_name
        }


class DocumentProcessor:
    """Main document processing pipeline."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vector_db = VectorDatabase()
        self.data_manager = DataManager()
    
    def process_scraped_content(self, contents: List[ScrapedContent]) -> int:
        """Process scraped content and add to vector database."""
        all_chunks = []
        
        for content in contents:
            logger.info(f"Processing: {content.title}")
            
            # Detect language
            language = self.preprocessor.detect_language(content.content)
            
            # Chunk the content
            chunks = self.preprocessor.chunk_text(content.content, content.title)
            
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    'text': chunk['text'],
                    'source_url': content.url,
                    'source_name': content.source,
                    'title': content.title,
                    'language': language,
                    'chunk_index': i,
                    'original_content_hash': content.content_hash,
                    'metadata': content.metadata
                }
                all_chunks.append(chunk_doc)
        
        # Add to vector database
        if all_chunks:
            self.vector_db.add_documents(all_chunks)
            logger.info(f"Processed {len(contents)} documents into {len(all_chunks)} chunks")
        
        return len(all_chunks)
    
    def process_latest_data(self) -> int:
        """Process the latest scraped data."""
        contents = self.data_manager.load_latest_data()
        
        if not contents:
            logger.warning("No scraped data found")
            return 0
        
        return self.process_scraped_content(contents)
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        return self.vector_db.search(query, k)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        return self.vector_db.get_stats()


def main():
    """Main processing function."""
    processor = DocumentProcessor()
    
    # Process latest scraped data
    chunk_count = processor.process_latest_data()
    
    if chunk_count > 0:
        logger.info(f"Successfully processed {chunk_count} text chunks")
        
        # Show database stats
        stats = processor.get_database_stats()
        logger.info(f"Database stats: {stats}")
    else:
        logger.warning("No data was processed")


if __name__ == "__main__":
    main()
