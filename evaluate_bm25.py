
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
import pandas as pd
import os

# Load test dataset
df = pd.read_csv("data/dataset_split/test.csv")

# Use 20% of the test set
df = df.sample(frac=0.20, random_state=42)

scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)

scores = []
processed = 0
missing = 0

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
        print("Missing:", syllabus_name)
        missing += 1
        continue

    with open(syllabus_path, encoding="ISO-8859-1") as f:
        text = f.read()

    # Paper's chunking settings
    chunk_size = 200
    overlap = 50
    step = chunk_size - overlap

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), step)
    ]

    if not chunks:
        continue

    # BM25 retrieval
    tokenized_chunks = [
        chunk.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    query_tokens = question.lower().split()

    # Search Baseline: retrieve top-1 chunk
    top1 = bm25.get_top_n(
        query_tokens,
        chunks,
        n=1
    )[0]

    # Calculate ROUGE-L F1
    score = scorer.score(
        reference,
        top1
    )["rougeL"].fmeasure

    scores.append(score)
    processed += 1

    if processed % 25 == 0:
        print(f"Processed {processed}/{len(df)}")

# Display results
print("\n==============================")
print("SEARCH BASELINE REPRODUCTION")
print("==============================")
print(f"Questions evaluated: {processed}")
print(f"Missing syllabi:     {missing}")

if scores:
    average_rouge = sum(scores) / len(scores)

    print(f"Our ROUGE-L F1:      {average_rouge:.4f}")
    print("Paper ROUGE-L F1:    0.118")
    print(f"Difference:          {average_rouge - 0.118:+.4f}")
else:
    print("No questions were evaluated.")

print("==============================")
