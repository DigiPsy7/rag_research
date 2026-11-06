import pandas as pd
import glob
from rank_bm25 import BM25Okapi

# Load test dataset
df = pd.read_csv("data/dataset_split/test.csv")

# Load syllabus text files
files = glob.glob("syllabi/syllabi_redacted/text/*.txt")

print("Dataset loaded:", len(df), "questions")
print("Syllabus files found:", len(files))

# Load one syllabus
syllabi = {}

for file in files:
    name = file.replace("\\", "/").split("/")[-1].replace(".txt", "")

    with open(file, "r", encoding="ISO-8859-1") as f:
        syllabi[name] = f.read()

# Take first question
row = df.iloc[0]

print("\nQuestion:")
print(row["question"])

print("\nSyllabus:")
print(row["syllabus_name"])

# Get the corresponding syllabus
text = syllabi[row["syllabus_name"]]

# Simple chunks
chunk_size = 1000
overlap = 200

chunks = []

start = 0
while start < len(text):
    chunks.append(text[start:start + chunk_size])
    start += chunk_size - overlap

print("\nNumber of chunks:", len(chunks))

# Tokenize
tokenized_chunks = [chunk.lower().split() for chunk in chunks]

# BM25
bm25 = BM25Okapi(tokenized_chunks)

query = row["question"].lower().split()

scores = bm25.get_scores(query)

# Get top 5
top_indices = scores.argsort()[::-1][:5]

print("\nTop 5 retrieved chunks:")

for rank, index in enumerate(top_indices, 1):
    print(f"\n--- Rank {rank} ---")
    print(chunks[index][:500])

print("\nBM25 retrieval test completed successfully!")