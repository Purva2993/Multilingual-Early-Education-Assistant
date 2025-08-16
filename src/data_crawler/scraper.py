"""
Data crawler and scraper for educational sources.
"""

import asyncio
import aiohttp
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, asdict
import time

from bs4 import BeautifulSoup
from newspaper import Article
import requests
from loguru import logger

from ..config import settings, load_education_sources


@dataclass
class ScrapedContent:
    """Data structure for scraped content."""
    title: str
    content: str
    url: str
    source: str
    language: str
    scraped_at: str
    content_hash: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScrapedContent':
        """Create from dictionary."""
        return cls(**data)


class EducationScraper:
    """Main scraper class for educational content."""
    
    def __init__(self):
        self.session = None
        self.sources_config = load_education_sources()
        self.scraped_urls = set()
        self.failed_urls = set()
        
    async def __aenter__(self):
        """Async context manager entry."""
        timeout = aiohttp.ClientTimeout(total=settings.REQUEST_TIMEOUT)
        connector = aiohttp.TCPConnector(limit=settings.MAX_CONCURRENT_REQUESTS)
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': settings.USER_AGENT,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _generate_content_hash(self, content: str) -> str:
        """Generate hash for content deduplication."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Filter out very short lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch content from a URL."""
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'html' in content_type:
                        return await response.text()
                    else:
                        logger.warning(f"Skipping non-HTML content: {url}")
                        return None
                else:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching {url}")
            self.failed_urls.add(url)
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            self.failed_urls.add(url)
            return None
    
    def _extract_content_bs4(self, html: str, url: str) -> Optional[ScrapedContent]:
        """Extract content using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text().strip() if title_elem else ""
            
            # Try to find main content
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main-content'])
            
            if not main_content:
                # Fall back to body
                main_content = soup.find('body')
            
            if not main_content:
                return None
            
            # Extract text content
            content = main_content.get_text(separator='\n', strip=True)
            content = self._clean_text(content)
            
            if len(content) < 100:  # Skip very short content
                return None
            
            # Extract metadata
            metadata = {}
            
            # Meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata['description'] = meta_desc.get('content', '')
            
            # Meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                metadata['keywords'] = meta_keywords.get('content', '')
            
            # Author
            author_meta = soup.find('meta', attrs={'name': 'author'})
            if author_meta:
                metadata['author'] = author_meta.get('content', '')
            
            # Publication date
            date_meta = soup.find('meta', attrs={'property': 'article:published_time'}) or \
                       soup.find('meta', attrs={'name': 'date'})
            if date_meta:
                metadata['published_date'] = date_meta.get('content', '')
            
            return ScrapedContent(
                title=title,
                content=content,
                url=url,
                source=urlparse(url).netloc,
                language='en',  # Default to English, can be detected later
                scraped_at=datetime.now(timezone.utc).isoformat(),
                content_hash=self._generate_content_hash(content),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return None
    
    def _extract_content_newspaper(self, url: str) -> Optional[ScrapedContent]:
        """Extract content using newspaper3k."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if not article.text or len(article.text) < 100:
                return None
            
            # Clean content
            content = self._clean_text(article.text)
            
            metadata = {
                'authors': article.authors,
                'publish_date': article.publish_date.isoformat() if article.publish_date else '',
                'top_image': article.top_image,
                'summary': article.summary
            }
            
            return ScrapedContent(
                title=article.title or "",
                content=content,
                url=url,
                source=urlparse(url).netloc,
                language='en',
                scraped_at=datetime.now(timezone.utc).isoformat(),
                content_hash=self._generate_content_hash(content),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error with newspaper extraction for {url}: {str(e)}")
            return None
    
    async def scrape_url(self, url: str) -> Optional[ScrapedContent]:
        """Scrape content from a single URL."""
        if url in self.scraped_urls or url in self.failed_urls:
            return None
        
        logger.info(f"Scraping: {url}")
        
        # Try async fetch first
        html = await self._fetch_url(url)
        if html:
            content = self._extract_content_bs4(html, url)
            if content:
                self.scraped_urls.add(url)
                return content
        
        # Fall back to newspaper3k
        try:
            content = self._extract_content_newspaper(url)
            if content:
                self.scraped_urls.add(url)
                return content
        except Exception as e:
            logger.error(f"Both extraction methods failed for {url}: {str(e)}")
        
        self.failed_urls.add(url)
        return None
    
    async def scrape_source(self, source_config: Dict[str, Any]) -> List[ScrapedContent]:
        """Scrape all pages from a source."""
        contents = []
        base_url = source_config['base_url']
        pages = source_config.get('pages', [])
        
        logger.info(f"Scraping source: {source_config['name']}")
        
        tasks = []
        for page in pages:
            full_url = urljoin(base_url, page)
            tasks.append(self.scrape_url(full_url))
            
            # Add delay between requests
            await asyncio.sleep(settings.SCRAPING_DELAY)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ScrapedContent):
                contents.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Scraping task failed: {str(result)}")
        
        return contents
    
    async def scrape_all_sources(self) -> List[ScrapedContent]:
        """Scrape all configured sources."""
        all_contents = []
        
        # Scrape each source category
        for category, sources in self.sources_config.items():
            if category == 'crawling_config':
                continue
                
            logger.info(f"Processing {category}...")
            
            for source in sources:
                try:
                    contents = await self.scrape_source(source)
                    all_contents.extend(contents)
                    logger.info(f"Scraped {len(contents)} documents from {source['name']}")
                    
                    # Delay between sources
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error scraping source {source['name']}: {str(e)}")
        
        return all_contents
    
    def save_scraped_data(self, contents: List[ScrapedContent], filename: Optional[str] = None) -> str:
        """Save scraped content to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scraped_education_data_{timestamp}.json"
        
        filepath = Path(settings.SCRAPED_DATA_DIR) / filename
        
        # Convert to dictionaries for JSON serialization
        data = {
            'scraped_at': datetime.now(timezone.utc).isoformat(),
            'total_documents': len(contents),
            'documents': [content.to_dict() for content in contents]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(contents)} documents to {filepath}")
        return str(filepath)


class DataManager:
    """Manages scraped data persistence and retrieval."""
    
    def __init__(self):
        self.data_dir = Path(settings.SCRAPED_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_latest_data(self) -> Optional[List[ScrapedContent]]:
        """Load the most recent scraped data."""
        json_files = list(self.data_dir.glob("scraped_education_data_*.json"))
        
        if not json_files:
            return None
        
        # Get the most recent file
        latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
        
        return self.load_data_file(latest_file)
    
    def load_data_file(self, filepath: Path) -> List[ScrapedContent]:
        """Load data from a specific JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = data.get('documents', [])
            return [ScrapedContent.from_dict(doc) for doc in documents]
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {str(e)}")
            return []
    
    def get_all_data_files(self) -> List[Path]:
        """Get all scraped data files."""
        return sorted(
            self.data_dir.glob("scraped_education_data_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
    
    def cleanup_old_files(self, keep_count: int = 5):
        """Remove old data files, keeping only the most recent ones."""
        files = self.get_all_data_files()
        
        if len(files) > keep_count:
            for file_to_remove in files[keep_count:]:
                try:
                    file_to_remove.unlink()
                    logger.info(f"Removed old data file: {file_to_remove}")
                except Exception as e:
                    logger.error(f"Error removing file {file_to_remove}: {str(e)}")


async def main():
    """Main scraping function."""
    async with EducationScraper() as scraper:
        contents = await scraper.scrape_all_sources()
        
        if contents:
            filepath = scraper.save_scraped_data(contents)
            logger.info(f"Scraping completed. Saved {len(contents)} documents to {filepath}")
            
            # Cleanup old files
            data_manager = DataManager()
            data_manager.cleanup_old_files()
        else:
            logger.warning("No content was scraped")


if __name__ == "__main__":
    asyncio.run(main())
