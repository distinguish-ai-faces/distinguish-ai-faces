"""
Tests for web scraping functionality.
"""
import pytest
import requests
from unittest.mock import Mock, patch
from ..scrape_faces import (
    ImageScraper,
    check_connection,
    download_image,
    validate_image
)

@pytest.fixture
def scraping_config():
    """Fixture for scraping configuration."""
    return {
        'sites': [
            {
                'name': 'thispersondoesnotexist',
                'url': 'https://thispersondoesnotexist.com',
                'image_selector': 'img#face',
                'rate_limit': 1.0,  # 1 request per second
                'requires_js': True
            },
            {
                'name': 'generated_photos',
                'url': 'https://generated.photos',
                'image_selector': '.photo-container img',
                'rate_limit': 0.5,  # 1 request per 2 seconds
                'requires_js': False
            }
        ],
        'headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        },
        'timeout': 10,
        'max_retries': 3
    }

@pytest.fixture
def mock_response():
    """Fixture for mocking HTTP responses."""
    mock = Mock()
    mock.status_code = 200
    mock.content = b"fake_image_content"
    mock.headers = {'Content-Type': 'image/jpeg'}
    return mock

def test_scraper_initialization(scraping_config):
    """Test scraper initialization with different site configs."""
    for site_config in scraping_config['sites']:
        scraper = ImageScraper(site_config)
        assert scraper.url == site_config['url']
        assert scraper.rate_limit == site_config['rate_limit']
        assert scraper.image_selector == site_config['image_selector']

@pytest.mark.parametrize("site_name", ["thispersondoesnotexist", "generated_photos"])
def test_connection_successful(scraping_config, mock_response, site_name):
    """Test successful connection to different data sources."""
    site_config = next(site for site in scraping_config['sites'] if site['name'] == site_name)
    
    with patch('requests.get', return_value=mock_response):
        assert check_connection(site_config['url']) is True

def test_connection_timeout(scraping_config):
    """Test connection timeout handling for all sites."""
    for site_config in scraping_config['sites']:
        with patch('requests.get', side_effect=requests.Timeout):
            with pytest.raises(requests.Timeout):
                check_connection(site_config['url'])

@pytest.mark.parametrize("status_code,expected", [
    (200, True),
    (429, False),  # Rate limit
    (403, False),  # Forbidden
    (404, False),  # Not found
    (500, False)   # Server error
])
def test_rate_limiting(scraping_config, status_code, expected):
    """Test rate limiting response handling."""
    mock_resp = Mock()
    mock_resp.status_code = status_code
    
    for site_config in scraping_config['sites']:
        scraper = ImageScraper(site_config)
        with patch('requests.get', return_value=mock_resp):
            assert scraper.check_rate_limit() == expected

def test_image_download_with_retry(mock_response, scraping_config):
    """Test image download with retry mechanism."""
    site_config = scraping_config['sites'][0]
    scraper = ImageScraper(site_config)
    
    # Simulate first failure, then success
    with patch('requests.get') as mock_get:
        mock_get.side_effect = [
            requests.RequestException("First attempt failed"),
            mock_response
        ]
        
        image_data = scraper.download_image_with_retry("https://example.com/image.jpg")
        assert image_data == mock_response.content
        assert mock_get.call_count == 2

def test_parallel_download(scraping_config):
    """Test parallel image download from multiple sources."""
    image_urls = [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
        "https://example.com/image3.jpg"
    ]
    
    with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map.return_value = [
            b"image1_data",
            b"image2_data",
            b"image3_data"
        ]
        
        scraper = ImageScraper(scraping_config['sites'][0])
        results = scraper.download_batch(image_urls, max_workers=3)
        assert len(results) == len(image_urls)
        assert all(result is not None for result in results)

def test_js_required_handling(scraping_config):
    """Test handling of sites requiring JavaScript."""
    for site_config in scraping_config['sites']:
        scraper = ImageScraper(site_config)
        if site_config['requires_js']:
            with patch('selenium.webdriver.Chrome') as mock_driver:
                mock_driver.return_value.page_source = "<html><img src='test.jpg'></html>"
                result = scraper.get_page_content()
                assert result is not None
        else:
            with patch('requests.get', return_value=Mock(text="<html></html>")):
                result = scraper.get_page_content()
                assert result is not None

@pytest.mark.parametrize("content_type,expected", [
    ('image/jpeg', True),
    ('image/png', True),
    ('image/webp', True),
    ('text/html', False),
    ('application/json', False)
])
def test_content_type_validation(content_type, expected):
    """Test validation of different content types."""
    mock_resp = Mock()
    mock_resp.headers = {'Content-Type': content_type}
    mock_resp.content = b"fake_content"
    
    assert validate_image(mock_resp) == expected

def test_proxy_support(scraping_config):
    """Test scraping through proxy."""
    proxy_config = {
        'http': 'http://proxy.example.com:8080',
        'https': 'https://proxy.example.com:8080'
    }
    
    site_config = scraping_config['sites'][0]
    scraper = ImageScraper(site_config)
    
    with patch('requests.get') as mock_get:
        scraper.download_image("https://example.com/image.jpg", proxies=proxy_config)
        mock_get.assert_called_once()
        assert mock_get.call_args[1]['proxies'] == proxy_config

def test_robot_txt_compliance(scraping_config):
    """Test compliance with robots.txt."""
    with patch('urllib.robotparser.RobotFileParser') as mock_parser:
        mock_parser.return_value.can_fetch.return_value = True
        
        for site_config in scraping_config['sites']:
            scraper = ImageScraper(site_config)
            assert scraper.check_robots_txt() is True 