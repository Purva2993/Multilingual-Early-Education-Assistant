"""
Main configuration file for the Multilingual AI Voice Assistant.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
import yaml
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Application Info
    APP_NAME: str = "Multilingual AI Voice Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Streamlit Settings
    STREAMLIT_HOST: str = "0.0.0.0"
    STREAMLIT_PORT: int = 8501
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./education_assistant.db"
    
    # Vector Database Settings
    VECTOR_DB_TYPE: str = "faiss"  # faiss or chroma
    VECTOR_DB_PATH: str = "./data/vector_db"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    
    # LLM Settings
    LLM_PROVIDER: str = "ollama"  # ollama, openai
    LLM_MODEL: str = "mistral:7b"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1000
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # OpenAI Settings (if using OpenAI)
    OPENAI_API_KEY: str = ""
    
    # Translation Settings
    TRANSLATION_MODEL: str = "Helsinki-NLP/opus-mt"
    DEFAULT_LANGUAGE: str = "en"
    
    # Voice Settings
    TTS_ENGINE: str = "coqui"  # coqui, gtts
    TTS_MODEL: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    VOICE_LANGUAGE: str = "en"
    
    # Speech Recognition Settings
    STT_ENGINE: str = "whisper"  # whisper, google
    WHISPER_MODEL: str = "base"
    
    # Scraping Settings
    SCRAPING_DELAY: float = 1.0
    MAX_CONCURRENT_REQUESTS: int = 5
    REQUEST_TIMEOUT: int = 30
    USER_AGENT: str = "Educational Assistant Bot 1.0"
    
    # Data Storage
    DATA_DIR: str = "./data"
    SCRAPED_DATA_DIR: str = "./data/scraped"
    AUDIO_DIR: str = "./data/audio"
    LOGS_DIR: str = "./logs"
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    # Crawling Schedule
    CRAWL_SCHEDULE: str = "daily"  # daily, weekly, monthly
    CRAWL_TIME: str = "02:00"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Load education sources configuration
def load_education_sources() -> Dict[str, Any]:
    """Load education sources from YAML configuration."""
    config_path = Path(__file__).parent / "config" / "education_sources.yaml"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    # Default configuration if file doesn't exist
    return {
        "government_sources": [
            {
                "name": "UNESCO",
                "base_url": "https://www.unesco.org",
                "pages": [
                    "/en/early-childhood-education",
                    "/en/education/early-childhood"
                ],
                "language": "en",
                "priority": "high"
            },
            {
                "name": "US Department of Education",
                "base_url": "https://www.ed.gov",
                "pages": [
                    "/early-learning",
                    "/about/offices/list/oese/pi/earlylearning"
                ],
                "language": "en",
                "priority": "high"
            }
        ],
        "ngo_sources": [
            {
                "name": "Zero to Three",
                "base_url": "https://www.zerotothree.org",
                "pages": [
                    "/resources/series/little-kids-big-feelings",
                    "/early-development"
                ],
                "language": "en",
                "priority": "medium"
            }
        ],
        "academic_sources": [
            {
                "name": "Early Childhood Research Quarterly",
                "base_url": "https://www.journals.elsevier.com/early-childhood-research-quarterly",
                "pages": ["/recent-articles"],
                "language": "en",
                "priority": "medium"
            }
        ]
    }


# Language configuration
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi"
}

# Translation model mappings
TRANSLATION_MODELS = {
    "en-es": "Helsinki-NLP/opus-mt-en-es",
    "es-en": "Helsinki-NLP/opus-mt-es-en",
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
    "en-it": "Helsinki-NLP/opus-mt-en-it",
    "it-en": "Helsinki-NLP/opus-mt-it-en",
    "en-pt": "Helsinki-NLP/opus-mt-en-roa",
    "pt-en": "Helsinki-NLP/opus-mt-roa-en",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",
}

# Create settings instance
settings = Settings()

# Ensure directories exist
for directory in [
    settings.DATA_DIR,
    settings.SCRAPED_DATA_DIR,
    settings.AUDIO_DIR,
    settings.LOGS_DIR,
    settings.VECTOR_DB_PATH
]:
    Path(directory).mkdir(parents=True, exist_ok=True)
