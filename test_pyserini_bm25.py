import os
import re
from rank_bm25 import BM25Okapi

SYLLABUS_DIR = "syllabi/syllabi_redacted/text"
FILENAME = "MUSIC-ED 500KU Syllabus S23.txt"

QUESTION = (
    "What are the names of the two websites, or technology platforms "
    "required to access course materials?"
)

filepath = os.path.join(SYLLABUS_DIR, FILENAME)

with open(filepath, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

# 200-character chunks with 50-character overlap
chunks = []

start = 0
while start < len(text):
    end = min(start + 200, len(text))
    chunk = text[start:end]

    if chunk.strip():
        chunks.append(chunk)

    start += 150

# Tokenization
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

tokenized_chunks = [tokenize(chunk) for chunk in chunks]

bm25 = BM25Okapi(tokenized_chunks)

query_tokens = tokenize(QUESTION)

scores = bm25.get_scores(query_tokens)

# Get top 5
top_indices = sorted(
    range(len(scores)),
    key=lambda i: scores[i],
    reverse=True
)[:5]

print("=" * 70)
print("QUESTION:")
print(QUESTION)

print("\nTOP 5 BM25 CHUNKS")
print("=" * 70)

for rank, index in enumerate(top_indices, start=1):
    print(f"\n--- Rank {rank} | Score: {scores[index]:.4f} ---")
    print(chunks[index])

    if "moodle" in chunks[index].lower() or "zoom" in chunks[index].lower():
        print("\n*** THIS CHUNK CONTAINS ZOOM OR MOODLE ***")