import os
import re
from pathlib import Path
from bs4 import BeautifulSoup

def parse_filing(filepath: str) -> str:
    """Parse SEC filing - extract EX-99.1 exhibit (earnings call transcript)."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read()

    # SEC full-submission files contain multiple documents
    # EX-99.1 is always the earnings call transcript exhibit
    extracted = None

    if '<DOCUMENT>' in raw:
        documents = raw.split('<DOCUMENT>')
        for doc in documents:
            lines = doc[:500]
            if '<TYPE>EX-99.1' in lines or 'TYPE>EX-99.1' in lines:
                extracted = doc
                break

    if extracted:
        raw = extracted

    # Parse HTML
    if '<html' in raw.lower() or '<body' in raw.lower():
        soup = BeautifulSoup(raw, 'html.parser')
        for tag in soup(['script', 'style', 'header', 'footer']):
            tag.decompose()
        text = soup.get_text(separator='\n')
    else:
        text = raw

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = '\n'.join(lines)

    return text


def is_transcript(text: str) -> bool:
    """Check if a filing actually contains earnings call transcript content."""
    transcript_markers = [
        'operator', 'conference call', 'earnings call',
        'q&a', 'question and answer', 'analyst',
        'prepared remarks', 'opening remarks'
    ]
    text_lower = text.lower()
    matches = sum(1 for marker in transcript_markers if marker in text_lower)
    
    # Also reject documents that are too long to be transcripts
    # Real transcripts are 10,000 - 80,000 characters
    if len(text) > 500000:
        return False
    
    return matches >= 1


def find_transcript_files(data_dir: str) -> list[dict]:
    filings = []
    base = Path(data_dir)

    for ticker_dir in base.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        filing_type_dir = ticker_dir / "8-K"
        if not filing_type_dir.exists():
            continue

        for filing_dir in filing_type_dir.iterdir():
            if not filing_dir.is_dir():
                continue
            date_str = filing_dir.name

            for file in filing_dir.iterdir():
                if file.suffix in ['.htm', '.html', '.txt']:
                    try:
                        text = parse_filing(str(file))
                        if is_transcript(text):  # only keep actual transcripts
                            filings.append({
                                "ticker": ticker,
                                "date": date_str,
                                "filepath": str(file)
                            })
                            break
                    except:
                        continue

    print(f"Found {len(filings)} transcript files (filtered from all 8-K filings)")
    return filings


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "../data/sec-edgar-filings")
    filings = find_transcript_files(data_dir)
    print(f"Found {len(filings)} filings\n")

    for f in filings[:2]:  # preview first 2
        print(f"Ticker: {f['ticker']} | Date: {f['date']}")
        text = parse_filing(f['filepath'])
        print(f"Text length: {len(text)} chars")
        print("Preview:")
        print(text[:500])
        print("-" * 60)

