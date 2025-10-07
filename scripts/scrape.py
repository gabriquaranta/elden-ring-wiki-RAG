#!/usr/bin/env python3
"""
elden ring wiki scraper

scrapes all relevant pages from the elden ring fextralife wiki and caches raw html.
"""

import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
from pathlib import Path


class EldenRingWikiScraper:
    def __init__(self, base_url="https://eldenring.wiki.fextralife.com", delay=1.0):
        self.base_url = base_url
        self.delay = delay  # delay between requests
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

        # create data dir
        self.data_dir = Path("data")
        self.raw_html_dir = self.data_dir / "raw_html"
        self.raw_html_dir.mkdir(parents=True, exist_ok=True)

    def get_page_urls(self):
        """discover all wiki page urls from the main page and category pages."""
        urls = set()

        # main wiki page
        print("discovering wiki pages...")
        try:
            response = self.session.get(self.base_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # find all wiki links
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("/"):
                    full_url = urljoin(self.base_url, href)
                    if self._is_wiki_page(full_url):
                        urls.add(full_url)

        except Exception as e:
            print(f"error discovering pages: {e}")
            return urls

        print(f"found {len(urls)} potential wiki pages")
        return urls

    def _is_wiki_page(self, url):
        """check if url is a valid wiki page (not category, file, etc.)."""
        parsed = urlparse(url)
        path = parsed.path.lower()

        # skip non-wiki pages
        if any(
            skip in path
            for skip in ["category:", "file:", "special:", "talk:", "template:"]
        ):
            return False

        # only include pages that look like articles
        if path.count("/") > 2:  # too many subdirs
            return False

        return True

    def scrape_page(self, url):
        """scrape a single page and return its content."""
        try:
            print(f"scraping: {url}")
            response = self.session.get(url)
            response.raise_for_status()

            # raw html
            filename = self._url_to_filename(url)
            filepath = self.raw_html_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)

            time.sleep(self.delay)
            return response.text

        except Exception as e:
            print(f"error scraping {url}: {e}")
            return None

    def _url_to_filename(self, url):
        """convert url to a safe filename."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            path = "index"

        # replace problematic characters
        safe_name = (
            path.replace("/", "_").replace(":", "_").replace("?", "_").replace("&", "_")
        )
        return f"{safe_name}.html"

    def scrape_all_pages(self):
        """scrape all discovered wiki pages."""
        urls = self.get_page_urls()
        scraped_data = []

        for i, url in enumerate(urls, 1):
            print(f"[{i}/{len(urls)}] processing: {url}")
            content = self.scrape_page(url)

            if content:
                scraped_data.append(
                    {
                        "url": url,
                        "filename": self._url_to_filename(url),
                        "scraped_at": time.time(),
                    }
                )

        # save metadata
        metadata_file = self.data_dir / "scraping_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(scraped_data, f, indent=2, ensure_ascii=False)

        print(f"scraping complete! processed {len(scraped_data)} pages.")
        return scraped_data


def main():
    scraper = EldenRingWikiScraper()
    scraper.scrape_all_pages()


if __name__ == "__main__":
    main()
