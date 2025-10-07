#!/usr/bin/env python3
"""
elden ring wiki scraper

scrapes all relevant pages from the elden ring fextralife wiki and caches raw html.

supports both single-page discovery (fast) and recursive crawling (comprehensive).
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

    def get_page_urls_recursive(self, max_depth=2):
        """recursively discover all wiki pages up to max_depth."""
        visited = set()
        to_visit = {self.base_url}
        urls = set()

        for depth in range(max_depth):
            print(f"crawling depth {depth + 1}/{max_depth}...")
            next_visit = set()

            for url in to_visit:
                if url in visited:
                    continue

                try:
                    response = self.session.get(url)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, "html.parser")

                    # extract links from this page
                    for link in soup.find_all("a", href=True):
                        href = link["href"]
                        if href.startswith("/"):
                            full_url = urljoin(url, href)
                            if (
                                self._is_wiki_page(full_url)
                                and full_url not in visited
                                and full_url not in urls
                            ):
                                urls.add(full_url)
                                if full_url not in visited:
                                    next_visit.add(full_url)

                    visited.add(url)
                    time.sleep(self.delay)

                except Exception as e:
                    print(f"error crawling {url}: {e}")

            to_visit = next_visit
            print(f"depth {depth + 1}: found {len(urls)} pages so far")

        print(f"recursive crawling complete! found {len(urls)} wiki pages")
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

    def scrape_all_pages(self, recursive=False, max_depth=2):
        """scrape all discovered wiki pages."""
        if recursive:
            urls = self.get_page_urls_recursive(max_depth)
        else:
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
    import argparse

    parser = argparse.ArgumentParser(description="scrape elden ring wiki pages")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="use recursive crawling to find more pages",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="maximum crawl depth for recursive mode (default: 2)",
    )

    args = parser.parse_args()

    scraper = EldenRingWikiScraper()

    if args.recursive:
        print(f"starting recursive crawl with max depth {args.depth}...")
        scraper.scrape_all_pages(recursive=True, max_depth=args.depth)
    else:
        print("starting single-page discovery...")
        scraper.scrape_all_pages(recursive=False)


if __name__ == "__main__":
    main()
