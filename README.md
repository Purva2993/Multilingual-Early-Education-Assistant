# Multilingual AI Voice Assistant

A comprehensive, open-source education search assistant that provides multilingual support for voice and text queries with trusted source verification.

## 🌟 Features

- **Multilingual Support**: Ask questions in 12+ languages and get responses in your preferred language
- **Voice Interface**: Speak your questions and hear the answers
- **Trusted Sources**: Information sourced from government, NGO, and academic institutions
- **Real-time Search**: Instant retrieval using advanced vector search
- **Open Source LLM**: Powered by Ollama with Mistral/LLaMA models
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Modern UI**: Streamlit frontend with chat interface
- **Docker Deployment**: Containerized for easy deployment
- **CI/CD Pipeline**: Automated testing and deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Streamlit UI  │    │   FastAPI API   │
│   (Education)   │    │   (Frontend)    │    │   (Backend)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Web Scraper    │    │  Voice Process  │    │  Query Process  │
│  (BeautifulSoup)│    │  (STT/TTS)      │    │  (Translation)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Vector Database │    │  LLM Integration│    │  Answer Engine  │
│ (FAISS/Chroma)  │    │   (Ollama)      │    │   (Generation)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional)
- Ollama (for LLM support)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd multilingual-ai-voice-assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configurations
   ```

4. **Install and start Ollama:**
   ```bash
   # Install Ollama (https://ollama.ai/)
   ollama pull mistral:7b
   ```

5. **Initialize the system:**
   ```bash
   # Crawl education data
   python -m src.data_crawler.scraper
   
   # Process and index data
   python -m src.preprocessing.indexer
   ```

6. **Start the services:**
   ```bash
   # Start FastAPI backend
   python -m src.api.main
   
   # In another terminal, start Streamlit frontend
   streamlit run src/frontend/streamlit_app.py
   ```

### Docker Deployment

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Frontend: http://localhost:8501
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 📊 Components

### Data Crawling & Scraping
- Automated crawling of trusted education websites
- Configurable sources via YAML
- Respect for robots.txt and rate limiting
- Content deduplication and quality filtering

### Preprocessing & Indexing
- Text cleaning and normalization
- Intelligent chunking (300-500 tokens)
- Multilingual embeddings with sentence-transformers
- Vector storage with FAISS

### Query Processing
- Language detection with langdetect
- Translation using MarianMT models
- Query optimization for better search results

### LLM Integration
- Ollama integration for local LLM hosting
- Configurable models (Mistral, LLaMA, etc.)
- Context-aware answer generation
- Source citation and verification

### Voice Processing
- Speech recognition with multiple engines
- Text-to-speech in multiple languages
- Audio file upload support
- Real-time voice interaction

## 🌐 Supported Languages

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- Arabic (ar)
- Hindi (hi)

## 🔧 Configuration

### Environment Variables

```bash
# Application Settings
APP_NAME="Multilingual AI Voice Assistant"
DEBUG=false

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Settings
STREAMLIT_HOST=0.0.0.0
STREAMLIT_PORT=8501

# Database Settings
DATABASE_URL=sqlite:///./education_assistant.db

# Vector Database
VECTOR_DB_TYPE=faiss
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM Settings
LLM_PROVIDER=ollama
LLM_MODEL=mistral:7b
OLLAMA_BASE_URL=http://localhost:11434

# Voice Settings
TTS_ENGINE=coqui
STT_ENGINE=whisper
```

### Education Sources Configuration

Edit `src/config/education_sources.yaml` to customize data sources:

```yaml
government_sources:
  - name: "UNESCO"
    base_url: "https://www.unesco.org"
    pages: ["/en/early-childhood-education"]
    language: "en"
    priority: "high"
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test modules
pytest tests/test_scraper.py
pytest tests/test_processor.py
```

## 📈 Monitoring

The application includes built-in monitoring:

- Health check endpoints
- Performance metrics
- Database statistics
- Error tracking with structured logging

## 🚢 Deployment

### Production Deployment

1. **Environment Setup:**
   ```bash
   # Set production environment variables
   export DEBUG=false
   export API_HOST=0.0.0.0
   export DATABASE_URL=postgresql://user:pass@host:5432/db
   ```

2. **Deploy with Docker:**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Set up reverse proxy (nginx):**
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location /api/ {
           proxy_pass http://localhost:8000/;
       }
       
       location / {
           proxy_pass http://localhost:8501/;
       }
   }
   ```

### Scaling

- Use Redis for caching and session storage
- Deploy multiple API instances behind a load balancer
- Use PostgreSQL for production database
- Implement horizontal scaling with Kubernetes

## 🔄 CI/CD Pipeline

The project includes GitHub Actions workflows for:

- **Testing**: Automated testing on push/PR
- **Building**: Docker image building and pushing
- **Deployment**: Automated deployment to staging/production
- **Security**: Dependency scanning and security checks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run code formatting
black src/
isort src/

# Run linting
flake8 src/
```

## 📝 API Documentation

### Query Endpoint

```http
POST /query
Content-Type: application/json

{
  "query": "What are the best practices for early childhood education?",
  "language": "en",
  "max_sources": 5,
  "include_audio": false
}
```

### Response

```json
{
  "id": "uuid",
  "query": "What are the best practices for early childhood education?",
  "answer": "Based on the provided sources, here are the key best practices...",
  "sources": [
    {
      "title": "Early Childhood Education Guidelines",
      "url": "https://unesco.org/...",
      "source_name": "UNESCO",
      "similarity_score": 0.95
    }
  ],
  "processing_time": 2.5,
  "generated_at": "2025-01-01T12:00:00Z"
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # Start Ollama service
   ollama serve
   ```

2. **Audio Issues**
   ```bash
   # Install system audio dependencies (Ubuntu)
   sudo apt-get install portaudio19-dev python3-pyaudio
   
   # Install ffmpeg for audio processing
   sudo apt-get install ffmpeg
   ```

3. **Memory Issues**
   ```bash
   # Reduce chunk size in config
   CHUNK_SIZE=200
   
   # Use smaller embedding model
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L12-v2
   ```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM hosting
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Streamlit](https://streamlit.io/) for the frontend
- Educational institutions providing open access to knowledge

## 📞 Support

For support, email support@example.com or join our [Discord community](https://discord.gg/example).

## 🗺️ Roadmap

- [ ] Additional LLM providers (OpenAI, Anthropic)
- [ ] Mobile app development
- [ ] Advanced voice features (conversation memory)
- [ ] Integration with more education platforms
- [ ] Multilingual model fine-tuning
- [ ] Enterprise features (SSO, analytics)

---

**Made with ❤️ for educators and parents worldwide**
