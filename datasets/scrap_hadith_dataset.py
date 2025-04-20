import requests
from bs4 import BeautifulSoup
from typing import List
import logging
import os
import json
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def is_valid_individual_url(url):
    """Check if the URL is a valid individual hadith page."""
    response = requests.get(url, allow_redirects=True)
    if response.status_code != 200:
        return False
    # Check if it redirects to a generic page (e.g., collection main page)
    if url.split('/')[-1] not in response.url:
        return False
    return True

def extract_collections(main_url: str) -> List[dict]:
    """Extract all collections from the main page of sunnah.com."""
    soup = BeautifulSoup(requests.get(main_url).content, 'html.parser')
    collections_div = soup.find('div', class_='collections')
    collections = []
    for collection_div in collections_div.find_all('div', class_='collection_title'): #collection_sep
        a_tag = collection_div.find('a')
        if a_tag:
            path = a_tag['href'][1:]  # Remove leading '/'
            display_name = a_tag.get_text(strip=True)
            collections.append({'display_name': display_name, 'path': path})
    
    return collections

class HadithScraper:
    """A class to scrape Hadiths from sunnah.com."""

    def __init__(self, base_url: str, data_dir: str):
        self.base_url = base_url
        self.data_dir = data_dir
        self.collection_path = base_url.split('/')[-1]
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        os.makedirs(self.data_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(self.data_dir, 'scraper.log'))
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def scrape_website(self, url: str) -> BeautifulSoup:
        """Scrape a website and return parsed content."""
        try:
            page = requests.get(url, timeout=10)
            page.raise_for_status()
            return BeautifulSoup(page.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error scraping {url}: {e}")
            return None

    def extract_books(self, soup: BeautifulSoup) -> List[dict]:
        """Extract book data from parsed HTML content."""
        if not soup:
            return []
        self.logger.info('Extracting book data from website...')
        books_containers = soup.find_all(class_='book_title')
        if not books_containers:
            return []
        books_data = []
        for container in books_containers:
            title_number = container.find(class_='title_number')
            if not title_number:
                continue
            book_number = title_number.get_text(strip=True)
            book_link = f"{self.base_url}/{book_number}"
            english_name = container.find(class_='english_book_name').get_text(strip=True) if container.find(class_='english_book_name') else 'Unknown'
            arabic_name = container.find(class_='arabic_book_name').get_text(strip=True) if container.find(class_='arabic_book_name') else 'غير معروف'
            books_data.append({
                'book_link': book_link,
                'book_number': book_number,
                'english_name': english_name,
                'arabic_name': arabic_name
            })
        self.logger.info(f'{len(books_data)} books extracted.')
        return books_data

    def extract_hadith_urls(self, book_url: str) -> List[str]:
        """Extract URLs of individual hadith pages from a book page."""
        soup = self.scrape_website(book_url)
        if not soup:
            return []
        hadith_titles = soup.find_all(class_='hadith_title')
        hadith_urls = []
        for title in hadith_titles:
            a_tag = title.find('a')
            if a_tag and 'href' in a_tag.attrs:
                hadith_urls.append('https://sunnah.com' + a_tag['href'])
        return hadith_urls

    def extract_hadiths(self, url: str) -> List[dict]:
        """Extract hadith data from a given URL."""
        soup = self.scrape_website(url)
        if not soup:
            return []
        hadith_containers = soup.find_all(class_='actualHadithContainer')
        hadiths_data = []
        for container in hadith_containers:
            hadith_en_text = container.find(class_='english_hadith_full').get_text(strip=True) if container.find(class_='english_hadith_full') else None
            hadith_ar_text = container.find(class_='arabic_hadith_full').get_text(strip=True) if container.find(class_='arabic_hadith_full') else None
            hadith_ref = container.find(class_='hadith_reference').get_text(strip=True) if container.find(class_='hadith_reference') else None
            grade = container.find(class_='english_grade').get_text(strip=True) if container.find(class_='english_grade') else None

            if hadith_ref:
                ref_parts = hadith_ref.split('In-book reference:')
                if len(ref_parts) > 1:
                    reference = ref_parts[0].strip().replace('Reference:', '').strip()
                    book_ref = ref_parts[1].strip()
                    book_ref_parts = book_ref.split(',')
                    if len(book_ref_parts) >= 2:
                        book_reference = book_ref_parts[0].strip()
                        hadith_number = book_ref_parts[1].strip().split('USC-MSA')[0].strip()
                    else:
                        book_reference = None
                        hadith_number = book_ref
                else:
                    reference = hadith_ref.strip().replace('Reference:', '').strip()
                    book_reference = None
                    hadith_number = reference
            else:
                reference = None
                book_reference = None
                hadith_number = None

            hadith_id = container.get('id')

            hadith_link = self.base_url + ':' + reference.split(' ')[-1]
            if not is_valid_individual_url(hadith_link):
                hadith_link = url + '#' + hadith_id if hadith_id else url

            hadiths_data.append({
                'english': hadith_en_text,
                'arabic': hadith_ar_text,
                'reference': reference,
                'book_reference': book_reference,
                'hadith_number': hadith_number,
                'grade': grade,
                'link': hadith_link
            })
        return hadiths_data

    def save_to_json(self, data: dict):
        """Save data to a JSON file based on whether it’s a book or collection."""
        if 'book_number' in data:
            filename = f"{data['book_number'].zfill(2)}_{data['english_name']}.json"
        else:
            filename = "hadiths.json"
        filepath = os.path.join(self.data_dir, filename)
        self.logger.info(f'Saving data to {filepath}...')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        self.logger.info(f'Data saved to {filepath}.')

    def scrape_collection(self):
        """Scrape all hadiths from the collection."""
        self.logger.info(f'Scraping collection: {self.collection_path}...')
        soup = self.scrape_website(self.base_url)
        if not soup:
            return
        books = self.extract_books(soup)
        if books:
            for book_data in tqdm(books, desc=f"Scraping {self.collection_path} books"):
                book_url = book_data['book_link']
                hadiths_data = self.extract_hadiths(book_url)
                if hadiths_data:
                    # Type 2: Hadiths are on the book page
                    book_data['hadith_data'] = hadiths_data
                else:
                    # Type 3: Hadiths are on individual pages
                    hadith_urls = self.extract_hadith_urls(book_url)
                    hadiths_data = []
                    for hadith_url in tqdm(hadith_urls, desc=f"Scraping hadiths for book {book_data['book_number']}", leave=False):
                        hadith_data = self.extract_hadiths(hadith_url)
                        if hadith_data:
                            hadiths_data.append(hadith_data[0])
                        time.sleep(0.5)  # Polite delay
                    book_data['hadith_data'] = hadiths_data
                self.save_to_json(book_data)
        else:
            # No books, scrape hadiths from main page
            hadiths_data = self.extract_hadiths(self.base_url)
            collection_data = {
                'collection_name': self.collection_path,
                'hadith_data': hadiths_data
            }
            self.save_to_json(collection_data)
        self.logger.info(f'Finished scraping {self.collection_path}.')

if __name__ == '__main__':
    main_url = 'https://sunnah.com'
    collections = extract_collections(main_url)
    for collection in tqdm(collections, desc="Scraping all collections"):
        collection_path = collection['path']
        collection_data_dir = os.path.join('./Hadith/sunnah_dot_com_data', collection_path)
        scraper = HadithScraper(base_url=f"{main_url}/{collection_path}", data_dir=collection_data_dir)
        scraper.scrape_collection()