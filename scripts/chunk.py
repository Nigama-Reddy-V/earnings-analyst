import re
import tiktoken
from typing import List

tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def split_by_speaker(text: str) -> List[dict]:
    """
    Split transcript text into speaker turns.
    Earnings calls follow patterns like:
    'OPERATOR:', 'TIM COOK:', 'LUCA MAESTRI:', 'ANALYST:'
    """
    # Pattern: ALL CAPS name followed by colon (common in earnings transcripts)
    speaker_pattern = re.compile(
        r'\n([A-Z][A-Z\s\.\-]{2,40}):\s*\n?', re.MULTILINE
    )

    segments = []
    matches = list(speaker_pattern.finditer(text))

    if len(matches) < 3:
        # Transcript doesn't have clear speaker labels
        # Fall back to paragraph chunking
        return split_by_paragraph(text)

    for i, match in enumerate(matches):
        speaker = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        if len(content) < 50:  # skip very short turns
            continue

        # Determine section
        speaker_lower = speaker.lower()
        if any(word in speaker_lower for word in ['operator', 'moderator']):
            section = 'operator'
        elif any(word in speaker_lower for word in ['analyst', 'research']):
            section = 'qa'
        else:
            section = 'management'

        segments.append({
            "speaker": speaker,
            "section": section,
            "text": content
        })

    return segments


def split_by_paragraph(text: str) -> List[dict]:
    """Fallback: split by paragraph if no speaker labels found."""
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
    return [{"speaker": "unknown", "section": "unknown", "text": p}
            for p in paragraphs]


def chunk_segments(
    segments: List[dict],
    ticker: str,
    date: str,
    max_tokens: int = 400,
    overlap_tokens: int = 50
) -> List[dict]:
    """
    Convert speaker segments into final chunks with metadata.
    Long segments get split further. Short ones stay as-is.
    """
    chunks = []
    chunk_id = 0

    for seg in segments:
        text = seg['text']
        token_count = count_tokens(text)

        if token_count <= max_tokens:
            # Segment fits in one chunk
            chunks.append({
                "chunk_id": f"{ticker}_{date}_{chunk_id}",
                "ticker": ticker,
                "date": date,
                "speaker": seg['speaker'],
                "section": seg['section'],
                "text": text,
                "token_count": token_count
            })
            chunk_id += 1

        else:
            # Split long segment into overlapping chunks
            words = text.split()
            current_chunk = []
            current_tokens = 0

            for word in words:
                current_chunk.append(word)
                current_tokens = count_tokens(' '.join(current_chunk))

                if current_tokens >= max_tokens:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        "chunk_id": f"{ticker}_{date}_{chunk_id}",
                        "ticker": ticker,
                        "date": date,
                        "speaker": seg['speaker'],
                        "section": seg['section'],
                        "text": chunk_text,
                        "token_count": current_tokens
                    })
                    chunk_id += 1

                    # Keep last N words for overlap
                    overlap_words = current_chunk[-15:]
                    current_chunk = overlap_words
                    current_tokens = count_tokens(' '.join(current_chunk))

            # Don't forget the last partial chunk
            if current_chunk and current_tokens > 50:
                chunks.append({
                    "chunk_id": f"{ticker}_{date}_{chunk_id}",
                    "ticker": ticker,
                    "date": date,
                    "speaker": seg['speaker'],
                    "section": seg['section'],
                    "text": ' '.join(current_chunk),
                    "token_count": current_tokens
                })
                chunk_id += 1

    return chunks


if __name__ == "__main__":
    # Quick test with dummy text
    sample = """
TIM COOK:
Good afternoon everyone. We're pleased to report another strong quarter.
iPhone revenue grew 6% year over year, driven by strong demand in emerging markets.
Our services business continues to be a key growth driver with 15% revenue growth.
We saw particular strength in the App Store and Apple Music segments.

LUCA MAESTRI:
Thank you Tim. Our gross margin came in at 44.5%, up 80 basis points sequentially.
This was driven primarily by favorable product mix and services growth.
Operating expenses were 14.2 billion, in line with our guidance.
We generated 29 billion in operating cash flow during the quarter.

ANALYST:
Can you give us more color on the gross margin outlook for next quarter?

LUCA MAESTRI:
We're guiding to gross margins between 45.5% and 46.5% for the December quarter.
This reflects continued services mix benefit partially offset by seasonal product costs.
"""

    segments = split_by_speaker(sample)
    chunks = chunk_segments(segments, "AAPL", "2024-09-30")

    print(f"Segments found: {len(segments)}")
    print(f"Chunks created: {len(chunks)}\n")
    for c in chunks:
        print(f"[{c['ticker']} | {c['speaker']} | {c['section']} | {c['token_count']} tokens]")
        print(c['text'][:150])
        print()