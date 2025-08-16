"""
LLM integration and answer generation module.
"""

import json
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime

from loguru import logger

from ..config import settings
from ..preprocessing.indexer import DocumentProcessor
from ..query_processing.processor import QueryProcessor


class OllamaLLM:
    """Ollama LLM integration."""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.LLM_MODEL
        self.temperature = settings.LLM_TEMPERATURE
        self.max_tokens = settings.LLM_MAX_TOKENS
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"Connected to Ollama. Available models: {model_names}")
                
                if self.model not in model_names:
                    logger.warning(f"Model {self.model} not found. Available: {model_names}")
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {str(e)}")
    
    def generate_response(self, prompt: str, context: str = "") -> Optional[str]:
        """Generate response using Ollama."""
        try:
            # Prepare the full prompt
            full_prompt = self._prepare_prompt(prompt, context)
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {str(e)}")
            return None
    
    def _prepare_prompt(self, query: str, context: str) -> str:
        """Prepare the prompt for the LLM."""
        system_prompt = """You are an expert early childhood education assistant. Your role is to provide accurate, helpful information about early childhood education, child development, and parenting based ONLY on the provided context documents.

IMPORTANT GUIDELINES:
- Only use information from the provided context documents
- If the context doesn't contain enough information to answer the question, say so clearly
- Always cite the sources when providing information
- Provide practical, evidence-based advice
- Be concise but comprehensive
- Focus on safety and developmental appropriateness
- If asked about medical concerns, recommend consulting healthcare professionals

Context Documents:
{context}

User Question: {query}

Please provide a helpful answer based on the context above. Include source references where appropriate."""
        
        return system_prompt.format(context=context, query=query)


class AnswerGenerator:
    """Main answer generation pipeline."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.query_processor = QueryProcessor()
        self.llm = OllamaLLM()
    
    def generate_answer(self, query: str, max_sources: int = 5) -> Dict[str, Any]:
        """
        Generate a comprehensive answer to a user query.
        
        Args:
            query: User's question
            max_sources: Maximum number of source documents to use
            
        Returns:
            Dictionary containing the answer and metadata
        """
        start_time = datetime.now()
        
        # Process the query
        query_info = self.query_processor.process_query(query)
        
        if query_info.get('error'):
            return {
                'query': query,
                'answer': "I'm sorry, I couldn't process your query. Please try rephrasing it.",
                'sources': [],
                'query_info': query_info,
                'processing_time': 0,
                'error': query_info['error']
            }
        
        # Search for relevant documents
        search_results = self.document_processor.search_documents(
            query_info['search_query'], 
            k=max_sources
        )
        
        if not search_results:
            return {
                'query': query,
                'answer': "I couldn't find any relevant information in my knowledge base for your question. Please try rephrasing or asking about a different topic related to early childhood education.",
                'sources': [],
                'query_info': query_info,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error': None
            }
        
        # Prepare context for LLM
        context = self._prepare_context(search_results)
        
        # Generate answer
        answer = self.llm.generate_response(query_info['search_query'], context)
        
        if not answer:
            return {
                'query': query,
                'answer': "I apologize, but I'm having trouble generating a response right now. Please try again later.",
                'sources': [],
                'query_info': query_info,
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error': "LLM generation failed"
            }
        
        # Translate answer back to original language if needed
        original_language = query_info['detected_language']
        if original_language != 'en':
            translated_answer = self.query_processor.translate_response(answer, original_language)
            if translated_answer:
                answer = translated_answer
        
        # Prepare sources information
        sources = self._prepare_sources(search_results)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'query': query,
            'answer': answer,
            'sources': sources,
            'query_info': query_info,
            'processing_time': processing_time,
            'error': None,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Generated answer for query: {query[:50]}... (took {processing_time:.2f}s)")
        return result
    
    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Prepare context string from search results."""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            source_info = f"Source {i}: {result.get('source_name', 'Unknown')} ({result.get('source_url', 'No URL')})"
            content = result.get('text', '')
            title = result.get('title', '')
            
            context_part = f"{source_info}\n"
            if title:
                context_part += f"Title: {title}\n"
            context_part += f"Content: {content}\n"
            context_part += f"Relevance Score: {result.get('similarity_score', 0):.3f}\n"
            context_part += "-" * 50 + "\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _prepare_sources(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare sources information for response."""
        sources = []
        
        for result in search_results:
            source = {
                'title': result.get('title', 'Untitled'),
                'url': result.get('source_url', ''),
                'source_name': result.get('source_name', 'Unknown Source'),
                'similarity_score': result.get('similarity_score', 0),
                'text_preview': result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', '')
            }
            sources.append(source)
        
        return sources
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        try:
            # Check database stats
            db_stats = self.document_processor.get_database_stats()
            
            # Check Ollama connection
            ollama_status = "connected"
            try:
                response = requests.get(f"{self.llm.base_url}/api/tags", timeout=5)
                if response.status_code != 200:
                    ollama_status = "disconnected"
            except:
                ollama_status = "disconnected"
            
            return {
                'database': db_stats,
                'ollama_status': ollama_status,
                'ollama_model': self.llm.model,
                'supported_languages': list(self.query_processor.get_supported_languages().keys()),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)}


def main():
    """Test answer generation."""
    generator = AnswerGenerator()
    
    # Test questions
    test_queries = [
        "What are the key developmental milestones for a 2-year-old?",
        "How can I help my child with language development?",
        "What are some good activities for preschoolers?",
        "¿Cómo puedo ayudar a mi hijo con el desarrollo del lenguaje?",  # Spanish
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        result = generator.generate_answer(query)
        
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        
        if result['sources']:
            print("\nSources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['title']} ({source['source_name']})")


if __name__ == "__main__":
    main()
