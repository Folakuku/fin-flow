import requests
from bs4 import BeautifulSoup
from typing import List

class WebScraper:
    """
    Web scraping component for financial news
    """
    def scrape_financial_news(self, stock_symbol: str) -> List[str]:
        """
        Scrape financial news related to the stock
        
        :param stock_symbol: Stock symbol to scrape news for
        :return: List of news headlines
        """
        url = f'https://finance.yahoo.com/quote/{stock_symbol}'
        try:
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            # Raise an exception for bad status codes
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract relevant news (this is a basic implementation)
            news_elements = soup.find_all('h3', class_='Mb(5px)')
            news = [elem.get_text() for elem in news_elements[:5]]
            
            return news
        except Exception as e:
            print(f"Error scraping news: {e}")
            return []
