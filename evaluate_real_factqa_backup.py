import os
import re
import random
import pandas as pd

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from google import genai


# ============================================================
# SETTINGS
# ============================================================

MODEL = "gemini-3.6-flash"

CHUNK_SIZE = 200
OVERLAP = 50

# First test only 5 questions
NUM_QUESTIONS = 5

RANDOM_STATE = 42


# ============================================================
# LOAD GEMINI
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in .env")

client = genai.Client(api_key=API_KEY)


# ============================================================
# LOAD DATASET
# ============================================================

TEST_FILE = "data/dataset_split/test.csv"
SYLLABUS_DIR = "syllabi/syllabi_redacted/text"

test_df = pd.read_csv(TEST_FILE)

print("Total test questions:", len(test_df))


# ============================================================
# SELECT SAME 20% STYLE SUBSET
# ============================================================

sample_size = int(len(test_df) * 0.20)

sample_df = test_df.sample(
    n=sample_size,
    random_state=RANDOM_STATE
)

# Only 5 questions for the first complete test
sample_df = sample_df.head(NUM_QUESTIONS).copy()

print("Questions used in this test:", len(sample_df))
print()


# ============================================================
# LOAD SYLLABI
# ============================================================

syllabi = {}

for filename in os.listdir(SYLLABUS_DIR):

    if filename.endswith(".txt"):

        path = os.path.join(SYLLABUS_DIR, filename)

        syllabus_name = os.path.splitext(filename)[0]

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            syllabi[syllabus_name] = f.read()


print("Syllabi loaded:", len(syllabi))
print()


# ============================================================
# CREATE 200/50 CHUNKS
# ============================================================

def create_chunks(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += CHUNK_SIZE - OVERLAP

    return chunks


# ============================================================
# BM25 RETRIEVAL
# ============================================================

def retrieve_top1(question, syllabus_text):

    chunks = create_chunks(syllabus_text)

    tokenized_chunks = [
        chunk.lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    question_tokens = question.lower().split()

    scores = bm25.get_scores(question_tokens)

    best_index = scores.argmax()

    return chunks[best_index]


# ============================================================
# GEMINI FACT-QA
# ============================================================

def ask_gemini(prompt):

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text


def get_score(answer_1, answer_2):

    prompt = f"""
You are evaluating factual consistency between two answers.

Answer 1:
{answer_1}

Answer 2:
{answer_2}

Follow these steps:

1. Identify all atomic factual claims in Answer 1.
2. For each claim, determine whether it is supported by Answer 2.
3. Count how many claims are supported.
4. Divide supported claims by total claims.

Return ONLY this format:

Score: X/Y

Do not provide any explanation.
"""

    result = ask_gemini(prompt)

    match = re.search(r"Score:\s*(\d+)\s*/\s*(\d+)", result)

    if not match:
        print("Could not parse Gemini response:")
        print(result)
        return None

    supported = int(match.group(1))
    total = int(match.group(2))

    if total == 0:
        return 0.0

    return supported / total


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(question, reference_answer, predicted_answer):

    # Reference -> Prediction
    # Measures recall
    recall = get_score(
        reference_answer,
        predicted_answer
    )

    # Prediction -> Reference
    # Measures precision
    precision = get_score(
        predicted_answer,
        reference_answer
    )

    if recall is not None and precision is not None:

        if recall + precision > 0:
            f1 = (
                2 * recall * precision
                / (recall + precision)
            )
        else:
            f1 = 0.0

    else:
        f1 = None

    return recall, precision, f1


# ============================================================
# MAIN EXPERIMENT
# ============================================================

results = []

for index, row in sample_df.iterrows():

    question = row["question"]
    reference_answer = row["answer"]
    syllabus_name = row["syllabus_name"]

    print("=" * 70)
    print("Question:")
    print(question)

    print("\nSyllabus:")
    print(syllabus_name)

    if syllabus_name not in syllabi:

        print("ERROR: syllabus not found")
        continue

    syllabus_text = syllabi[syllabus_name]

    # BM25 retrieves top-1 chunk
    predicted_answer = retrieve_top1(
        question,
        syllabus_text
    )

    print("\nBM25 predicted answer:")
    print(predicted_answer)

    print("\nReference answer:")
    print(reference_answer)

    print("\nRunning Gemini Fact-QA...")

    recall, precision, f1 = evaluate_question(
        question,
        reference_answer,
        predicted_answer
    )

    print("\nGemini results:")

    print("Recall:", recall)
    print("Precision:", precision)
    print("F1:", f1)

    results.append({
        "id": row["id"],
        "question": question,
        "syllabus_name": syllabus_name,
        "reference_answer": reference_answer,
        "predicted_answer": predicted_answer,
        "recall": recall,
        "precision": precision,
        "f1": f1
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "factqa_5_questions_results.csv",
    index=False
)


# ============================================================
# OVERALL RESULT
# ============================================================

if len(results_df) > 0:

    print("\n")
    print("=" * 70)
    print("FINAL 5-QUESTION RESULT")
    print("=" * 70)

    print(
        "Average Recall:",
        results_df["recall"].mean()
    )

    print(
        "Average Precision:",
        results_df["precision"].mean()
    )

    print(
        "Average F1:",
        results_df["f1"].mean()
    )

    print("\nResults saved to:")
    print("factqa_5_questions_results.csv")