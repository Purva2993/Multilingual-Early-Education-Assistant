"""
FastAPI backend for the Multilingual AI Voice Assistant.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from loguru import logger

from ..config import settings
from ..llm_integration.generator import AnswerGenerator
from ..voice_processing.voice_interface import VoiceInterface
from ..data_crawler.scraper import EducationScraper, DataManager
from ..preprocessing.indexer import DocumentProcessor


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    language: Optional[str] = None
    max_sources: int = 5
    include_audio: bool = False


class QueryResponse(BaseModel):
    id: str
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    query_info: Dict[str, Any]
    processing_time: float
    audio_url: Optional[str] = None
    generated_at: str
    error: Optional[str] = None


class SystemStatusResponse(BaseModel):
    status: str
    database: Dict[str, Any]
    ollama_status: str
    supported_languages: List[str]
    timestamp: str


class CrawlRequest(BaseModel):
    force_recrawl: bool = False


class CrawlResponse(BaseModel):
    status: str
    message: str
    documents_processed: int
    chunks_created: int
    timestamp: str


# Initialize FastAPI app
app = FastAPI(
    title="Multilingual AI Voice Assistant",
    description="A comprehensive education search assistant with multilingual voice and text support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
answer_generator = None
voice_interface = None
data_manager = None
document_processor = None

# Storage for async tasks and responses
response_cache = {}
background_tasks_status = {}


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global answer_generator, voice_interface, data_manager, document_processor
    
    logger.info("Initializing Multilingual AI Voice Assistant...")
    
    try:
        # Initialize components
        answer_generator = AnswerGenerator()
        voice_interface = VoiceInterface()
        data_manager = DataManager()
        document_processor = DocumentProcessor()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multilingual AI Voice Assistant API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status."""
    try:
        status_info = answer_generator.get_system_status()
        
        return SystemStatusResponse(
            status="operational",
            database=status_info.get('database', {}),
            ollama_status=status_info.get('ollama_status', 'unknown'),
            supported_languages=status_info.get('supported_languages', []),
            timestamp=status_info.get('timestamp', datetime.now().isoformat())
        )
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a text or voice query."""
    try:
        query_id = str(uuid.uuid4())
        
        # Generate answer
        result = answer_generator.generate_answer(
            request.query,
            max_sources=request.max_sources
        )
        
        # Generate audio if requested
        audio_url = None
        if request.include_audio and result['answer']:
            detected_language = result['query_info'].get('detected_language', 'en')
            tts_result = voice_interface.speak_text(result['answer'], detected_language)
            
            if tts_result['success']:
                audio_url = f"/audio/{query_id}.wav"
                # Store audio path for later retrieval
                response_cache[query_id] = tts_result['audio_path']
        
        response = QueryResponse(
            id=query_id,
            query=result['query'],
            answer=result['answer'],
            sources=result['sources'],
            query_info=result['query_info'],
            processing_time=result['processing_time'],
            audio_url=audio_url,
            generated_at=result['generated_at'],
            error=result.get('error')
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice-query")
async def process_voice_query(audio_file: UploadFile = File(...)):
    """Process a voice query from uploaded audio file."""
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/{audio_file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await audio_file.read()
            buffer.write(content)
        
        # Transcribe audio
        transcription = voice_interface.speech_recognizer.transcribe_audio_file(temp_path)
        
        if not transcription:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        # Process as text query
        request = QueryRequest(query=transcription, include_audio=True)
        response = await process_query(request)
        
        # Add transcription info
        response.query_info['transcribed_from_audio'] = True
        response.query_info['original_filename'] = audio_file.filename
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing voice query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{query_id}.wav")
async def get_audio(query_id: str):
    """Retrieve generated audio file."""
    if query_id not in response_cache:
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    audio_path = response_cache[query_id]
    return FileResponse(audio_path, media_type="audio/wav")


@app.post("/crawl", response_model=CrawlResponse)
async def trigger_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Trigger data crawling and processing."""
    try:
        task_id = str(uuid.uuid4())
        
        # Add background task
        background_tasks.add_task(
            run_crawl_and_process,
            task_id,
            request.force_recrawl
        )
        
        background_tasks_status[task_id] = {
            'status': 'started',
            'started_at': datetime.now().isoformat()
        }
        
        return CrawlResponse(
            status="started",
            message=f"Crawling task started with ID: {task_id}",
            documents_processed=0,
            chunks_created=0,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error triggering crawl: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/crawl-status/{task_id}")
async def get_crawl_status(task_id: str):
    """Get status of a crawling task."""
    if task_id not in background_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks_status[task_id]


@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages."""
    return answer_generator.query_processor.get_supported_languages()


@app.delete("/cache")
async def clear_cache():
    """Clear response cache and old audio files."""
    try:
        # Clear response cache
        response_cache.clear()
        
        # Clean up old audio files
        voice_interface.cleanup_old_audio_files()
        
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_crawl_and_process(task_id: str, force_recrawl: bool = False):
    """Background task for crawling and processing data."""
    try:
        background_tasks_status[task_id] = {
            'status': 'crawling',
            'started_at': background_tasks_status[task_id]['started_at'],
            'updated_at': datetime.now().isoformat()
        }
        
        # Run crawling
        async with EducationScraper() as scraper:
            contents = await scraper.scrape_all_sources()
            
            if contents:
                filepath = scraper.save_scraped_data(contents)
                
                # Process the data
                background_tasks_status[task_id]['status'] = 'processing'
                background_tasks_status[task_id]['updated_at'] = datetime.now().isoformat()
                
                chunk_count = document_processor.process_scraped_content(contents)
                
                # Update status
                background_tasks_status[task_id] = {
                    'status': 'completed',
                    'started_at': background_tasks_status[task_id]['started_at'],
                    'completed_at': datetime.now().isoformat(),
                    'documents_processed': len(contents),
                    'chunks_created': chunk_count,
                    'data_file': filepath
                }
                
                logger.info(f"Crawl task {task_id} completed: {len(contents)} docs, {chunk_count} chunks")
            else:
                background_tasks_status[task_id] = {
                    'status': 'failed',
                    'started_at': background_tasks_status[task_id]['started_at'],
                    'failed_at': datetime.now().isoformat(),
                    'error': 'No content was scraped'
                }
                
    except Exception as e:
        logger.error(f"Crawl task {task_id} failed: {str(e)}")
        background_tasks_status[task_id] = {
            'status': 'failed',
            'started_at': background_tasks_status[task_id].get('started_at'),
            'failed_at': datetime.now().isoformat(),
            'error': str(e)
        }


def run_server():
    """Run the FastAPI server."""
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()
