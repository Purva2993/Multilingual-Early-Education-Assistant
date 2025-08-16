"""
Main application runner for the Multilingual AI Voice Assistant.
"""

import asyncio
import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

from src.config import settings
from src.data_crawler.scraper import main as scraper_main
from src.preprocessing.indexer import main as indexer_main
from src.api.main import run_server
import subprocess
import signal
import time
from loguru import logger


def setup_logging():
    """Setup application logging."""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File logging
    logger.add(
        f"{settings.LOGS_DIR}/app.log",
        level="INFO",
        rotation="1 day",
        retention="30 days",
        compression="gzip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


async def crawl_data():
    """Crawl education data from sources."""
    logger.info("Starting data crawling...")
    try:
        await scraper_main()
        logger.info("Data crawling completed successfully")
        return True
    except Exception as e:
        logger.error(f"Data crawling failed: {str(e)}")
        return False


def process_data():
    """Process and index scraped data."""
    logger.info("Starting data processing and indexing...")
    try:
        indexer_main()
        logger.info("Data processing completed successfully")
        return True
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        return False


def check_ollama():
    """Check if Ollama is running and model is available."""
    import requests
    
    try:
        # Check if Ollama is running
        response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if settings.LLM_MODEL in model_names:
                logger.info(f"✅ Ollama is running with model {settings.LLM_MODEL}")
                return True
            else:
                logger.warning(f"⚠️  Ollama is running but model {settings.LLM_MODEL} not found")
                logger.info(f"Available models: {model_names}")
                logger.info(f"Pull the model with: ollama pull {settings.LLM_MODEL}")
                return False
        else:
            logger.error(f"❌ Ollama API returned status {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Could not connect to Ollama: {str(e)}")
        logger.info("Make sure Ollama is installed and running:")
        logger.info("1. Install Ollama from https://ollama.ai/")
        logger.info("2. Start Ollama: ollama serve")
        logger.info(f"3. Pull model: ollama pull {settings.LLM_MODEL}")
        return False


def run_api():
    """Run the FastAPI server."""
    logger.info("Starting FastAPI server...")
    run_server()


def run_streamlit():
    """Run the Streamlit frontend."""
    logger.info("Starting Streamlit frontend...")
    
    cmd = [
        "streamlit", "run", 
        "src/frontend/streamlit_app.py",
        "--server.address", settings.STREAMLIT_HOST,
        "--server.port", str(settings.STREAMLIT_PORT),
        "--server.headless", "true",
        "--server.enableCORS", "false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit: {str(e)}")
        sys.exit(1)


def run_full_system():
    """Run the complete system (API + Frontend)."""
    import multiprocessing
    import time
    
    logger.info("🚀 Starting Multilingual AI Voice Assistant...")
    
    # Check prerequisites
    if not check_ollama():
        logger.error("Ollama check failed. Please ensure Ollama is properly set up.")
        sys.exit(1)
    
    # Start API server in a separate process
    api_process = multiprocessing.Process(target=run_api, name="API-Server")
    api_process.start()
    
    # Wait for API to start up
    logger.info("Waiting for API server to start...")
    time.sleep(5)
    
    # Check if API is running
    import requests
    try:
        response = requests.get(f"http://{settings.API_HOST}:{settings.API_PORT}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API server started successfully")
        else:
            logger.error("❌ API server health check failed")
            api_process.terminate()
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Could not connect to API server: {str(e)}")
        api_process.terminate()
        sys.exit(1)
    
    # Start Streamlit frontend
    logger.info("Starting Streamlit frontend...")
    
    def signal_handler(signum, frame):
        logger.info("Shutting down...")
        api_process.terminate()
        api_process.join()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        run_streamlit()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        logger.info("Terminating API server...")
        api_process.terminate()
        api_process.join()


async def setup_system():
    """Setup the system with initial data."""
    logger.info("🔧 Setting up Multilingual AI Voice Assistant...")
    
    # Check Ollama
    if not check_ollama():
        logger.error("Please set up Ollama before continuing")
        return False
    
    # Crawl data
    logger.info("📥 Crawling education data...")
    if not await crawl_data():
        logger.error("Failed to crawl data")
        return False
    
    # Process data
    logger.info("⚙️ Processing and indexing data...")
    if not process_data():
        logger.error("Failed to process data")
        return False
    
    logger.info("✅ System setup completed successfully!")
    logger.info("You can now start the system with: python app.py run")
    return True


def main():
    """Main application entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Multilingual AI Voice Assistant")
    parser.add_argument(
        "command",
        choices=["setup", "crawl", "process", "api", "frontend", "run"],
        help="Command to execute"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force operation (e.g., re-crawl data)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.command == "setup":
            asyncio.run(setup_system())
        
        elif args.command == "crawl":
            asyncio.run(crawl_data())
        
        elif args.command == "process":
            process_data()
        
        elif args.command == "api":
            if not check_ollama():
                logger.error("Ollama check failed")
                sys.exit(1)
            run_api()
        
        elif args.command == "frontend":
            run_streamlit()
        
        elif args.command == "run":
            run_full_system()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
