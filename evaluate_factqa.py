import os
import re
import pandas as pd
from dotenv import load_dotenv
from google import genai

# Load API key from .env
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in .env")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.6-flash"


def ask_gemini(prompt):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )
    return response.text


def fact_qa(reference_answer, predicted_answer):
    prompt = f"""
You are evaluating the factual consistency of two answers.

Answer 1 (reference answer):
{reference_answer}

Answer 2 (predicted answer):
{predicted_answer}

Follow these steps:

1. Break Answer 1 into atomic factual claims.
2. For each atomic claim, determine whether it is supported by Answer 2.
3. Calculate:

Recall = number of reference claims supported by Answer 2
         / total number of reference claims

Precision = number of claims in Answer 2 supported by Answer 1
           / total number of claims in Answer 2

F1 = harmonic mean of Precision and Recall.

Return ONLY:

Recall: <value>
Precision: <value>
F1: <value>

Use values between 0 and 1.
"""

    result = ask_gemini(prompt)

    print("\nGemini evaluation:")
    print(result)

    return result


# Load SyllabusQA test dataset
test_path = "data/dataset_split/test.csv"
test = pd.read_csv(test_path)

# Use ONE question for the first test
row = test.iloc[0]

print("Question:")
print(row["question"])

print("\nReference answer:")
print(row["answer"])

# Temporary predicted answer:
# For the first API test, we deliberately use the reference answer itself.
# This lets us verify that the Fact-QA judge works correctly.
predicted_answer = row["answer"]

print("\nRunning Gemini Fact-QA test...")

fact_qa(
    reference_answer=row["answer"],
    predicted_answer=predicted_answer
)