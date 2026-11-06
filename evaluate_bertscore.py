from rank_bm25 import BM25Okapi
from bert_score import score
import pandas as pd
import os

# Load test dataset
df = pd.read_csv("data/dataset_split/test.csv")

# Same 20% subset approach
df = df.sample(frac=0.20, random_state=42)

references = []
candidates = []

processed = 0

for _, row in df.iterrows():

    syllabus_name = row["syllabus_name"]
    question = row["question"]
    reference = row["answer"]

    syllabus_path = os.path.join(
        "syllabi",
        "syllabi_redacted",
        "text",
        syllabus_name + ".txt"
    )

    if not os.path.exists(syllabus_path):
        continue

    with open(syllabus_path, encoding="ISO-8859-1") as f:
        text = f.read()

    # Paper's Search Baseline
    chunk_size = 200
    overlap = 50
    step = chunk_size - overlap

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), step)
    ]

    if not chunks:
        continue

    # BM25
    tokenized_chunks = [
        chunk.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    query_tokens = question.lower().split()

    # Top-1
    top1 = bm25.get_top_n(
        query_tokens,
        chunks,
        n=1
    )[0]

    references.append(reference)
    candidates.append(top1)

    processed += 1

    if processed % 25 == 0:
        print(f"Processed {processed}/{len(df)}")


print("\nCalculating BERTScore...")

# BERTScore
P, R, F1 = score(
    candidates,
    references,
    lang="en",
    verbose=True
)

average_f1 = F1.mean().item()

print("\n==============================")
print("BERTSCORE REPRODUCTION")
print("==============================")
print(f"Questions evaluated: {processed}")
print(f"Our BERTScore F1:     {average_f1:.4f}")
print(f"Paper BERTScore F1:   0.477")
print(f"Difference:           {average_f1 - 0.477:+.4f}")
print("==============================")