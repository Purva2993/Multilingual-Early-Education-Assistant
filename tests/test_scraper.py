"""
Test suite for the scraper module.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from src.data_crawler.scraper import EducationScraper, ScrapedContent, DataManager


class TestScrapedContent:
    """Test ScrapedContent data structure."""
    
    def test_scraped_content_creation(self):
        """Test creating ScrapedContent object."""
        content = ScrapedContent(
            title="Test Title",
            content="Test content",
            url="https://example.com",
            source="example.com",
            language="en",
            scraped_at=datetime.now().isoformat(),
            content_hash="abc123",
            metadata={"author": "Test Author"}
        )
        
        assert content.title == "Test Title"
        assert content.content == "Test content"
        assert content.url == "https://example.com"
        assert content.source == "example.com"
        assert content.language == "en"
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        content = ScrapedContent(
            title="Test",
            content="Content",
            url="https://example.com",
            source="example.com",
            language="en",
            scraped_at=datetime.now().isoformat(),
            content_hash="abc123",
            metadata={}
        )
        
        result = content.to_dict()
        assert isinstance(result, dict)
        assert result['title'] == "Test"
        assert result['content'] == "Content"
    
    def test_from_dict(self):
        """Test creating from dictionary."""
        data = {
            'title': "Test",
            'content': "Content",
            'url': "https://example.com",
            'source': "example.com",
            'language': "en",
            'scraped_at': datetime.now().isoformat(),
            'content_hash': "abc123",
            'metadata': {}
        }
        
        content = ScrapedContent.from_dict(data)
        assert content.title == "Test"
        assert content.content == "Content"


class TestEducationScraper:
    """Test EducationScraper class."""
    
    @pytest.fixture
    def scraper(self):
        """Create scraper instance for testing."""
        return EducationScraper()
    
    def test_init(self, scraper):
        """Test scraper initialization."""
        assert scraper.session is None
        assert scraper.scraped_urls == set()
        assert scraper.failed_urls == set()
    
    def test_generate_content_hash(self, scraper):
        """Test content hash generation."""
        content = "Test content"
        hash1 = scraper._generate_content_hash(content)
        hash2 = scraper._generate_content_hash(content)
        hash3 = scraper._generate_content_hash("Different content")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 32  # MD5 hash length
    
    def test_clean_text(self, scraper):
        """Test text cleaning."""
        dirty_text = "  Line 1  \n  \n  Line 2  \n\n  Short  \n  Line 3  "
        cleaned = scraper._clean_text(dirty_text)
        
        lines = cleaned.split('\n')
        assert "Line 1" in lines
        assert "Line 2" in lines
        assert "Line 3" in lines
        assert "Short" not in lines  # Should be filtered out
    
    @pytest.mark.asyncio
    async def test_fetch_url_success(self, scraper):
        """Test successful URL fetching."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.text = AsyncMock(return_value="<html>Test</html>")
        
        mock_session = Mock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock()
        
        scraper.session = mock_session
        
        result = await scraper._fetch_url("https://example.com")
        assert result == "<html>Test</html>"
    
    @pytest.mark.asyncio
    async def test_fetch_url_failure(self, scraper):
        """Test URL fetching failure."""
        mock_response = Mock()
        mock_response.status = 404
        
        mock_session = Mock()
        mock_session.get = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock()
        
        scraper.session = mock_session
        
        result = await scraper._fetch_url("https://example.com")
        assert result is None
        assert "https://example.com" in scraper.failed_urls
    
    def test_extract_content_bs4(self, scraper):
        """Test content extraction with BeautifulSoup."""
        html = """
        <html>
            <head><title>Test Title</title></head>
            <body>
                <main>
                    <h1>Main Heading</h1>
                    <p>This is a test paragraph with enough content to pass the minimum length requirement.</p>
                    <p>Another paragraph with more content to ensure proper extraction.</p>
                </main>
            </body>
        </html>
        """
        
        result = scraper._extract_content_bs4(html, "https://example.com")
        
        assert result is not None
        assert result.title == "Test Title"
        assert "Main Heading" in result.content
        assert "test paragraph" in result.content
        assert result.url == "https://example.com"
        assert result.source == "example.com"


class TestDataManager:
    """Test DataManager class."""
    
    @pytest.fixture
    def data_manager(self, tmp_path):
        """Create data manager with temporary directory."""
        with patch('src.data_crawler.scraper.settings') as mock_settings:
            mock_settings.SCRAPED_DATA_DIR = str(tmp_path)
            return DataManager()
    
    def test_init(self, data_manager, tmp_path):
        """Test data manager initialization."""
        assert data_manager.data_dir == tmp_path
        assert data_manager.data_dir.exists()
    
    def test_load_latest_data_no_files(self, data_manager):
        """Test loading data when no files exist."""
        result = data_manager.load_latest_data()
        assert result is None
    
    def test_load_data_file(self, data_manager, tmp_path):
        """Test loading data from specific file."""
        # Create test data file
        test_data = {
            'scraped_at': datetime.now().isoformat(),
            'total_documents': 1,
            'documents': [{
                'title': 'Test',
                'content': 'Test content',
                'url': 'https://example.com',
                'source': 'example.com',
                'language': 'en',
                'scraped_at': datetime.now().isoformat(),
                'content_hash': 'abc123',
                'metadata': {}
            }]
        }
        
        import json
        test_file = tmp_path / "test_data.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        
        result = data_manager.load_data_file(test_file)
        assert len(result) == 1
        assert result[0].title == 'Test'
        assert result[0].content == 'Test content'


@pytest.mark.integration
class TestIntegration:
    """Integration tests for scraper components."""
    
    @pytest.mark.asyncio
    async def test_full_scraping_pipeline(self):
        """Test complete scraping pipeline."""
        # This would be an integration test that requires actual network access
        # or mocked HTTP responses for the full pipeline
        pass
    
    def test_data_persistence(self, tmp_path):
        """Test data saving and loading."""
        # Create test scraped content
        content = ScrapedContent(
            title="Test Document",
            content="This is test content for persistence testing.",
            url="https://example.com/test",
            source="example.com",
            language="en",
            scraped_at=datetime.now().isoformat(),
            content_hash="test123",
            metadata={"test": True}
        )
        
        # Mock settings for temporary directory
        with patch('src.data_crawler.scraper.settings') as mock_settings:
            mock_settings.SCRAPED_DATA_DIR = str(tmp_path)
            
            scraper = EducationScraper()
            filename = scraper.save_scraped_data([content])
            
            # Verify file was created
            assert tmp_path.joinpath(filename.split('/')[-1]).exists()
            
            # Load and verify data
            data_manager = DataManager()
            loaded_data = data_manager.load_latest_data()
            
            assert len(loaded_data) == 1
            assert loaded_data[0].title == "Test Document"
            assert loaded_data[0].content == "This is test content for persistence testing."


if __name__ == "__main__":
    pytest.main([__file__])
