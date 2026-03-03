"""
Download CMS Medicare Benefit Policy Manual chapters from public URLs.
"""

import os
import sys
import httpx
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# CMS Medicare Benefit Policy Manual chapters to download
# Pattern: https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c{NN}.pdf
CMS_CHAPTERS = [
    {
        "chapter": "chapter_01",
        "number": "01",
        "description": "Inpatient Hospital Care",
    },
    {
        "chapter": "chapter_15",
        "number": "15",
        "description": "Covered Medical and Other Health Services",
    },
]

CMS_URL_PATTERN = "https://www.cms.gov/Regulations-and-Guidance/Guidance/Manuals/Downloads/bp102c{number}.pdf"

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "pdfs"


def download_chapter(chapter_info: dict, output_dir: Path) -> Path | None:
    """Download a single CMS chapter PDF."""
    url = CMS_URL_PATTERN.format(number=chapter_info["number"])
    filename = f"{chapter_info['chapter']}.pdf"
    output_path = output_dir / filename

    if output_path.exists():
        print(f"  [skip] {filename} already exists")
        return output_path

    print(f"  Downloading {filename} ({chapter_info['description']})...")
    print(f"    URL: {url}")

    try:
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            response = client.get(url)
            response.raise_for_status()

            output_path.write_bytes(response.content)
            size_kb = len(response.content) / 1024
            print(f"    Saved {size_kb:.1f} KB → {output_path}")
            return output_path

    except httpx.HTTPStatusError as e:
        print(f"    ERROR: HTTP {e.response.status_code} for {url}")
        return None
    except httpx.RequestError as e:
        print(f"    ERROR: {e}")
        return None


def main():
    print("CMS Medicare Policy Manual — PDF Downloader")
    print("=" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    downloaded = []
    failed = []

    for chapter in CMS_CHAPTERS:
        result = download_chapter(chapter, OUTPUT_DIR)
        if result:
            downloaded.append(result)
        else:
            failed.append(chapter["chapter"])

    print(f"\nResults: {len(downloaded)} downloaded, {len(failed)} failed")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
        print("  Note: CMS URLs may change. Check https://www.cms.gov/manuals/Downloads/ for current links.")

    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
