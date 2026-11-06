import os
import re
import pandas as pd
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from google import genai

# ============================================================
# 1. GEMINI SETUP
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.6-flash"


# ============================================================
# 2. TOKENIZATION
# ============================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# ============================================================
# 3. LOAD SYLLABI
# ============================================================

def load_syllabi():
    syllabus_dir = "syllabi/syllabi_redacted/text"

    syllabi = {}

    for filename in os.listdir(syllabus_dir):

        if filename.endswith(".txt"):

            path = os.path.join(syllabus_dir, filename)

            with open(
                path,
                "r",
                encoding="utf-8",
                errors="replace"
            ) as f:

                text = f.read()

            syllabus_name = filename[:-4]

            syllabi[syllabus_name] = text

    print("Syllabi loaded:", len(syllabi))

    return syllabi


# ============================================================
# 4. CREATE 200-CHARACTER CHUNKS
# ============================================================

def create_chunks(text, chunk_size=200, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# 5. BM25 TOP-1 RETRIEVAL
# ============================================================

def retrieve_top1(question, syllabus_text):

    chunks = create_chunks(
        syllabus_text,
        chunk_size=200,
        overlap=50
    )

    tokenized_chunks = [
        tokenize(chunk)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    query_tokens = tokenize(question)

    scores = bm25.get_scores(query_tokens)

    best_index = scores.argmax()

    return chunks[best_index], scores[best_index]


# ============================================================
# 6. FACT-QA PROMPT
# ============================================================

def factqa_prompt(question, answer_1, answer_2):

    return f"""
You are evaluating factual consistency between two answers.

Question:
{question}

Answer 1:
{answer_1}

Answer 2:
{answer_2}

Follow these steps:

1. Identify all atomic factual claims in Answer 1.
2. For each claim, determine whether it is supported by Answer 2.
3. Count the number of supported claims.
4. Count the total number of claims in Answer 1.

Then output your result exactly in this format:

Score: X/Y

Do not omit the Score line.
"""


# ============================================================
# 7. GEMINI FACT-QA EVALUATION
# ============================================================

def evaluate_direction(question, answer_1, answer_2):

    prompt = factqa_prompt(
        question,
        answer_1,
        answer_2
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
        "temperature": 0
        }
        
    )

    text = response.text.strip()

    print("\nGemini response:")
    print(text)

    match = re.search(
        r"Score\s*:\s*(\d+)\s*/\s*(\d+)",
        text,
        re.IGNORECASE
    )

    if not match:

        print("WARNING: Could not parse Score X/Y")

        return None, None, text

    supported = int(match.group(1))
    total = int(match.group(2))

    if total == 0:
        return None, None, text

    score = supported / total

    return score, (supported, total), text


# ============================================================
# 8. MAIN EXPERIMENT
# ============================================================

print("=" * 70)
print("SYLLABUSQA APPROXIMATE FACT-QA REPRODUCTION")
print("=" * 70)

# Load dataset
test_path = "data/dataset_split/test.csv"

test_df = pd.read_csv(test_path)

print("\nTotal test questions:", len(test_df))

# Same 20% sampling strategy used earlier
sample_df = test_df.sample(
    frac=0.20,
    random_state=42
).head(5)

print("Questions used:", len(sample_df))

# Load syllabi
syllabi = load_syllabi()

results = []


# ============================================================
# 9. PROCESS EACH QUESTION
# ============================================================

for i, row in sample_df.iterrows():

    question = row["question"]
    reference = row["answer"]
    syllabus_name = row["syllabus_name"]

    print("\n")
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    print("\nSyllabus:")
    print(syllabus_name)

    # Find syllabus
    syllabus_text = syllabi.get(syllabus_name)

    if syllabus_text is None:

        print("WARNING: Syllabus not found")

        continue

    # BM25 retrieval
    predicted, retrieval_score = retrieve_top1(
        question,
        syllabus_text
    )

    print("\nBM25 TOP-1 RETRIEVED TEXT:")
    print("-" * 70)
    print(predicted)

    print("\nBM25 score:", retrieval_score)

    print("\nREFERENCE ANSWER:")
    print("-" * 70)
    print(reference)

    # --------------------------------------------------------
    # Precision:
    # Predicted answer claims supported by reference
    # --------------------------------------------------------

    precision_score, precision_raw, precision_response = \
        evaluate_direction(
            question,
            predicted,
            reference
        )

    # --------------------------------------------------------
    # Recall:
    # Reference answer claims supported by prediction
    # --------------------------------------------------------

    recall_score, recall_raw, recall_response = \
        evaluate_direction(
            question,
            reference,
            predicted
        )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if (
        precision_score is not None
        and recall_score is not None
        and precision_score + recall_score > 0
    ):

        f1 = (
            2 * precision_score * recall_score
            / (precision_score + recall_score)
        )

    else:

        f1 = None

    print("\nRESULT")
    print("-" * 70)
    print("Precision:", precision_score)
    print("Recall:", recall_score)
    print("F1:", f1)

    results.append({

        "id": row["id"],
        "question": question,
        "syllabus_name": syllabus_name,
        "reference_answer": reference,
        "retrieved_answer": predicted,
        "retrieval_score": retrieval_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1

    })


# ============================================================
# 10. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = "factqa_5_questions_results.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\n")
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(output_file)

if len(results_df) > 0:

    print("\nAverage Precision:",
          results_df["precision"].mean())

    print("Average Recall:",
          results_df["recall"].mean())

    print("Average F1:",
          results_df["f1"].mean())