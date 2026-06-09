import glob
import os
import re

DOCUMENTS_DIR = "documents"
CHUNK_SIZE = 300
OVERLAP = 50


def clean_text(text):
    """Collapse runs of whitespace into single spaces and strip the ends."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split text into overlapping chunks using a sliding window."""
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def get_chunks(documents_dir=DOCUMENTS_DIR):
    """Load, clean, and chunk every .txt file.

    Returns a list of dicts, each with:
      - text: the chunk string
      - source: the source filename (e.g. "abiodun_robert.txt")
      - chunk_index: the chunk's position within its source file (0-based)
    """
    paths = sorted(glob.glob(os.path.join(documents_dir, "*.txt")))
    records = []

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        cleaned = clean_text(raw)
        source = os.path.basename(path)
        for i, chunk in enumerate(chunk_text(cleaned)):
            records.append({"text": chunk, "source": source, "chunk_index": i})

    return records


def main():
    records = get_chunks()

    counts = {}
    for r in records:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    for source in sorted(counts):
        print(f"{source}: {counts[source]} chunks")

    print(f"\nTotal chunks: {len(records)}")


if __name__ == "__main__":
    main()
