"""
Query processing module for handling user input and language detection.
"""

import re
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

from langdetect import detect, DetectorFactory
from transformers import MarianMTModel, MarianTokenizer
import torch
from loguru import logger

from ..config import settings, SUPPORTED_LANGUAGES, TRANSLATION_MODELS


class LanguageDetector:
    """Language detection utilities."""
    
    def __init__(self):
        # Set seed for consistent results
        DetectorFactory.seed = 0
        self.confidence_threshold = 0.8
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of input text.
        
        Returns:
            Tuple of (language_code, confidence)
        """
        if not text or len(text.strip()) < 3:
            return 'en', 0.5
        
        try:
            # Clean text for better detection
            cleaned_text = self._clean_text_for_detection(text)
            
            if len(cleaned_text) < 3:
                return 'en', 0.5
            
            detected_lang = detect(cleaned_text)
            
            # Validate against supported languages
            if detected_lang in SUPPORTED_LANGUAGES:
                return detected_lang, 0.9  # High confidence for supported languages
            else:
                # Default to English for unsupported languages
                logger.warning(f"Detected unsupported language: {detected_lang}, defaulting to English")
                return 'en', 0.7
                
        except Exception as e:
            logger.error(f"Language detection failed: {str(e)}")
            return 'en', 0.5
    
    def _clean_text_for_detection(self, text: str) -> str:
        """Clean text for better language detection."""
        # Remove URLs, emails, and special characters
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def is_supported_language(self, lang_code: str) -> bool:
        """Check if language is supported."""
        return lang_code in SUPPORTED_LANGUAGES


class TextTranslator:
    """Text translation using MarianMT models."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _get_translation_model_name(self, source_lang: str, target_lang: str) -> Optional[str]:
        """Get the model name for translation between two languages."""
        model_key = f"{source_lang}-{target_lang}"
        
        if model_key in TRANSLATION_MODELS:
            return TRANSLATION_MODELS[model_key]
        
        # Try reverse direction
        reverse_key = f"{target_lang}-{source_lang}"
        if reverse_key in TRANSLATION_MODELS:
            return TRANSLATION_MODELS[reverse_key]
        
        return None
    
    def _load_model(self, model_name: str) -> Tuple[MarianMTModel, MarianTokenizer]:
        """Load translation model and tokenizer."""
        if model_name not in self.models:
            try:
                logger.info(f"Loading translation model: {model_name}")
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                model.to(self.device)
                
                self.tokenizers[model_name] = tokenizer
                self.models[model_name] = model
                
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load translation model {model_name}: {str(e)}")
                raise
        
        return self.models[model_name], self.tokenizers[model_name]
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text or None if translation fails
        """
        if not text or source_lang == target_lang:
            return text
        
        # Get model name
        model_name = self._get_translation_model_name(source_lang, target_lang)
        if not model_name:
            logger.warning(f"No translation model available for {source_lang} -> {target_lang}")
            return None
        
        try:
            # Load model
            model, tokenizer = self._load_model(model_name)
            
            # Handle reverse translation if needed
            reverse_key = f"{target_lang}-{source_lang}"
            if model_name == TRANSLATION_MODELS.get(reverse_key) and source_lang != 'en':
                # This is a reverse model, we need to adjust
                actual_source, actual_target = target_lang, source_lang
            else:
                actual_source, actual_target = source_lang, target_lang
            
            # Prepare input
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode output
            translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            logger.info(f"Translated ({actual_source} -> {actual_target}): {text[:50]}... -> {translated_text[:50]}...")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return None
    
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English."""
        if source_lang == 'en':
            return text
        
        translated = self.translate(text, source_lang, 'en')
        return translated if translated else text
    
    def translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate text from English to target language."""
        if target_lang == 'en':
            return text
        
        translated = self.translate(text, 'en', target_lang)
        return translated if translated else text


class QueryProcessor:
    """Main query processing pipeline."""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.translator = TextTranslator()
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize query text."""
        if not query:
            return ""
        
        # Basic cleaning
        query = query.strip()
        
        # Remove excessive whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove leading/trailing punctuation
        query = query.strip('.,!?;:')
        
        return query
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and prepare it for search.
        
        Args:
            query: Raw user query
            
        Returns:
            Dictionary containing processed query information
        """
        # Clean the query
        cleaned_query = self._clean_query(query)
        
        if not cleaned_query:
            return {
                'original_query': query,
                'cleaned_query': '',
                'detected_language': 'en',
                'language_confidence': 0.0,
                'translated_query': '',
                'search_query': '',
                'error': 'Empty query'
            }
        
        # Detect language
        detected_lang, confidence = self.language_detector.detect_language(cleaned_query)
        
        # Translate to English if needed
        search_query = cleaned_query
        translated_query = ""
        
        if detected_lang != 'en':
            translated_query = self.translator.translate_to_english(cleaned_query, detected_lang)
            if translated_query:
                search_query = translated_query
            else:
                logger.warning(f"Translation failed for query: {cleaned_query}")
        
        # Prepare result
        result = {
            'original_query': query,
            'cleaned_query': cleaned_query,
            'detected_language': detected_lang,
            'language_name': SUPPORTED_LANGUAGES.get(detected_lang, 'Unknown'),
            'language_confidence': confidence,
            'translated_query': translated_query,
            'search_query': search_query,
            'processed_at': datetime.now().isoformat(),
            'error': None
        }
        
        logger.info(f"Processed query: {query} -> {search_query} (lang: {detected_lang})")
        return result
    
    def translate_response(self, response: str, target_language: str) -> str:
        """Translate response back to user's language."""
        if target_language == 'en' or not response:
            return response
        
        translated = self.translator.translate_from_english(response, target_language)
        return translated if translated else response
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages."""
        return SUPPORTED_LANGUAGES.copy()


def main():
    """Test query processing."""
    processor = QueryProcessor()
    
    # Test queries in different languages
    test_queries = [
        "What are the best practices for early childhood education?",
        "¿Cuáles son las mejores prácticas para la educación infantil temprana?",
        "Quelles sont les meilleures pratiques pour l'éducation de la petite enfance?",
        "Was sind die besten Praktiken für die frühkindliche Bildung?",
    ]
    
    for query in test_queries:
        print(f"\nTesting query: {query}")
        result = processor.process_query(query)
        print(f"Detected language: {result['language_name']} ({result['detected_language']})")
        print(f"Search query: {result['search_query']}")
        
        if result['translated_query']:
            print(f"Translation: {result['translated_query']}")


if __name__ == "__main__":
    main()
