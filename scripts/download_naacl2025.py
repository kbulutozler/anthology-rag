#!/usr/bin/env python3
"""
Script to download NAACL 2025 volumes bibtex files and paper PDFs.
"""

import os
import re
import sys
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from llm_agent.config import DATA_DIR

BASE_URL = "https://aclanthology.org"


def fetch_url(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""


def download_file(url: str, dest: str, mode: str = 'wb') -> bool:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def main():
    print("Fetching NAACL 2025 event page...")
    html = fetch_url(f"{BASE_URL}/events/naacl-2025/")
    if not html:
        sys.exit(1)
    soup = BeautifulSoup(html, 'html.parser')
    # Find all bibtex links for volumes
    bib_links = soup.find_all('a', href=re.compile(r'^/volumes/2025\..*\.bib$'))
    if not bib_links:
        print("No bibtex links found on the event page.")
        return

    for link in bib_links:
        href = link['href']
        full_bib_url = BASE_URL + href
        match = re.search(r'/volumes/(2025\..*?)\.bib$', href)
        if not match:
            continue
        volume_id = match.group(1)         # e.g., '2025.naacl-long'
        slug = volume_id.replace('.', '-')  # e.g., '2025-naacl-long'

        # Download bibtex
        bib_dir = DATA_DIR / "bibtex" / slug
        bib_dir.mkdir(parents=True, exist_ok=True)
        bib_dest = str(bib_dir / f"{slug}.bib")
        print(f"Downloading bibtex for {volume_id}...")
        if download_file(full_bib_url, bib_dest):
            print(f"Saved bibtex to {bib_dest}")

        # Fetch volume page to find papers
        vol_url = f"{BASE_URL}/volumes/{volume_id}"
        print(f"Fetching volume page {vol_url}...")
        vol_html = fetch_url(vol_url)
        if not vol_html:
            continue
        vol_soup = BeautifulSoup(vol_html, 'html.parser')
        # Find paper links like '/2025.naacl-long.1' or '/2025.naacl-long.1/'
        pattern = re.compile(rf'^/{re.escape(volume_id)}\.(\d+)(?:/)?$')
        paper_links = [a['href'] for a in vol_soup.find_all('a', href=pattern)]
        print(f"Found {len(paper_links)} papers for volume {volume_id}")

        # Download each paper PDF
        for href_p in paper_links:
            m = pattern.match(href_p)
            if not m:
                continue
            paper_idx = m.group(1)
            paper_id = f"{volume_id}.{paper_idx}"
            pdf_url = f"{BASE_URL}/{paper_id}.pdf"
            pdf_dir = DATA_DIR / "pdf" / slug
            pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf_dest = str(pdf_dir / f"{paper_idx}.pdf")
            # check if pdf exists, continue if so
            if Path(pdf_dest).exists():
                print(f"PDF for {paper_id} already exists, skipping...")
                continue
            print(f"Downloading PDF for paper {paper_id}...")
            if download_file(pdf_url, pdf_dest):
                print(f"Saved PDF to {pdf_dest}")
            time.sleep(0.1)

    print("All volumes processed.")


if __name__ == '__main__':
    main() 