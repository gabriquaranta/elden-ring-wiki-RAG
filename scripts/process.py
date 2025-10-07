#!/usr/bin/env python3
"""
Elden Ring Wiki Data Processor

Reads raw HTML files, extracts clean text content, and structures it as JSON.
"""

import json
import re
from pathlib import Path
from bs4 import BeautifulSoup


class EldenRingWikiProcessor:
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_html_dir = self.data_dir / "raw_html"
        self.cleaned_data_file = self.data_dir / "cleaned_data.json"

    def load_metadata(self):
        """Load scraping metadata to get list of pages."""
        metadata_file = self.data_dir / "scraping_metadata.json"
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def clean_html_content(self, html_content, url):
        """Extract clean text content from HTML, removing navigation, ads, etc."""
        soup = BeautifulSoup(html_content, "html.parser")

        # remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # remove common non-content elements
        selectors_to_remove = [
            "nav",
            "header",
            "footer",
            ".navbar",
            ".nav",
            ".menu",
            ".sidebar",
            ".advertisement",
            ".ads",
            ".comments",
            ".comment",
            "#comments",
            ".footer",
            ".header",
            ".navigation",
            ".social",
            ".share",
            ".breadcrumb",
            ".breadcrumbs",
            ".search",
            ".login",
            ".signup",
        ]

        for selector in selectors_to_remove:
            for element in soup.select(selector):
                element.decompose()

        # try multiple strategies to find content
        content = self._find_content_area(soup)

        # title
        title = self.extract_title(soup, url)

        # text content
        text_content = self.extract_text_content(content)

        return {"url": url, "title": title, "content": text_content}

    def _find_content_area(self, soup):
        """Find the main content area using multiple strategies."""
        # strategy 1: common content containers
        content_selectors = [
            "main",
            ".content",
            "#content",
            ".wiki-content",
            ".article-content",
            "article",
            ".main-content",
            ".page-content",
            ".entry-content",
            "#main",
            ".post-content",
        ]

        for selector in content_selectors:
            content = soup.select_one(selector)
            if content and len(content.get_text(strip=True)) > 200:
                return content

        # strat 2: largest text block
        all_text_blocks = []
        for element in soup.find_all(["div", "section", "article", "p"]):
            text = element.get_text(strip=True)
            if len(text) > 100:  # Only consider substantial text blocks
                all_text_blocks.append((element, len(text)))

        if all_text_blocks:
            # element with the most text
            all_text_blocks.sort(key=lambda x: x[1], reverse=True)
            return all_text_blocks[0][0]

        # strat 3: body with some cleaning
        body = soup.find("body")
        if body:
            # obvious non-content
            for element in body.find_all(
                ["nav", "header", "footer", "aside", ".comments"]
            ):
                element.decompose()
            return body

        return soup  # ultimate fallback

    def extract_title(self, soup, url):
        """Extract the page title."""
        # different title sources
        title_sources = [
            lambda: soup.find("h1").get_text().strip() if soup.find("h1") else None,
            lambda: (
                soup.find("title").get_text().strip() if soup.find("title") else None
            ),
            lambda: (
                soup.find(class_=re.compile(r"title", re.I)).get_text().strip()
                if soup.find(class_=re.compile(r"title", re.I))
                else None
            ),
        ]

        for title_func in title_sources:
            try:
                title = title_func()
                if title:
                    return title
            except:
                continue

        # fallback extract from URL
        url_path = url.split("/")[-1].replace("+", " ")
        return url_path

    def extract_text_content(self, element):
        """Extract clean text content from HTML element."""
        # get text while preserving some structure
        text = element.get_text(separator="\n", strip=True)

        # clean up the text
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            # skip very short or empty lines
            if len(line) > 10:
                lines.append(line)

        # double newlines to preserve paragraph structure
        return "\n\n".join(lines)

    def process_all_pages(self):
        """Process all scraped pages and create cleaned dataset."""
        metadata = self.load_metadata()
        cleaned_data = []

        print(f"Processing {len(metadata)} pages...")

        for i, page_info in enumerate(metadata, 1):
            filename = page_info["filename"]
            url = page_info["url"]

            print(f"[{i}/{len(metadata)}] Processing: {filename}")

            try:
                # read raw HTML
                html_file = self.raw_html_dir / filename
                with open(html_file, "r", encoding="utf-8") as f:
                    html_content = f.read()

                # clean and structure the content
                cleaned_page = self.clean_html_content(html_content, url)
                cleaned_data.append(cleaned_page)

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

        # save cleaned data
        with open(self.cleaned_data_file, "w", encoding="utf-8") as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        print(
            f"Processing complete! Saved {len(cleaned_data)} cleaned pages to {self.cleaned_data_file}"
        )

        # create a summary
        self.create_summary(cleaned_data)

        return cleaned_data

    def create_summary(self, cleaned_data):
        """Create a summary of the cleaned dataset."""
        summary = {
            "total_pages": len(cleaned_data),
            "total_content_length": sum(len(page["content"]) for page in cleaned_data),
            "average_content_length": (
                sum(len(page["content"]) for page in cleaned_data) / len(cleaned_data)
                if cleaned_data
                else 0
            ),
            "sample_pages": (
                cleaned_data[:3] if len(cleaned_data) >= 3 else cleaned_data
            ),
        }

        summary_file = self.data_dir / "cleaning_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"Summary saved to {summary_file}")


def main():
    processor = EldenRingWikiProcessor()
    processor.process_all_pages()


if __name__ == "__main__":
    main()
