"""
CorpusQA Evaluation Script

This script evaluates model-generated answers against ground truth using
an LLM-as-judge approach with ORM (Output Reward Model) evaluation.

Usage:
    python eval.py --input_file model_output.jsonl --model deepseek-v3

Author: CorpusQA Team
License: MIT
"""

import os
import json
import re
import time
from openai import OpenAI
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Helper: Extract JSON content from possible Markdown code blocks
def extract_json(content):
    """
    Extract JSON content from Markdown code blocks.

    Args:
        content (str): Raw content that may contain JSON in code blocks

    Returns:
        str: Extracted JSON string
    """
    match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return re.sub(r'`{3}.*?`{3}', '', content, flags=re.DOTALL).strip()

# Helper: Extract answer after 'The answer is: '
def extract_answer(response):
    """
    Extract the final answer from model response.
    Looks for the pattern 'The answer is: <answer>' or returns the full response.

    Args:
        response (str): Model's response text

    Returns:
        str: Extracted answer
    """
    match = re.search(r'The answer is: (.*)', response)
    if match:
        return match.group(1).strip()
    return response.strip()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Evaluate model-generated answers (extract answers first)")
parser.add_argument("--input_file", type=str, default="gemini-2.5-flash_financial_en_set1.jsonl", help="The jsonl filename in the runs directory")
parser.add_argument("--model", type=str, default="deepseek-v3", help="Model name")
args = parser.parse_args()
max_workers = 32  # Can be adjusted according to API rate limits


# --- 1. Configuration ---
input_file = args.input_file
input_path = os.path.join("runs", input_file)

# Automatically infer level and corpus
filename_without_ext = input_file.replace('.jsonl', '')
parts = filename_without_ext.split('_')
model = parts[0]
setting = parts[1] if len(parts) > 1 else ''
level = parts[2] if len(parts) > 2 else ''
corpus = '_'.join(parts[3:]) if len(parts) > 3 else ''

# --- 2. Load Model Answers ---
answers = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        answers.append(json.loads(line))

# --- 3. No longer read the gold file, directly use the 'answer' field of each item as the gold standard

# --- 4. Initialize OpenAI Client ---
# Load API key from environment variable
token = os.getenv("DASHSCOPE_API_KEY")
if not token:
    raise ValueError("DASHSCOPE_API_KEY environment variable not set.")

client = OpenAI(
    api_key=token,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# --- 5. Set Up Evaluation File ---
eval_dir = "evals"
os.makedirs(eval_dir, exist_ok=True)

eval_filename = f"{filename_without_ext}_eval.jsonl"
eval_path = os.path.join(eval_dir, eval_filename)

# --- 6. Resume Support: Track Processed IDs and Stats ---
processed_ids = set()
correct_count = 0
total = 0

# Track domain-specific statistics
domain_stats = {
    'financial_zh': {'correct': 0, 'total': 0},
    'financial_en': {'correct': 0, 'total': 0},
    'education_en': {'correct': 0, 'total': 0},
    'real_estate_en': {'correct': 0, 'total': 0}
}

def get_domain_from_id(question_id):
    """
    Extract domain from question ID prefix.
    Expected formats: financial_zh_*, financial_en_*, education_en_*, real_estate_en_*
    """
    if question_id.startswith('financial_zh'):
        return 'financial_zh'
    elif question_id.startswith('financial_en'):
        return 'financial_en'
    elif question_id.startswith('education_en'):
        return 'education_en'
    elif question_id.startswith('real_estate_en'):
        return 'real_estate_en'
    else:
        return None

# Read existing results if file exists
if os.path.exists(eval_path):
    with open(eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                q_id = data.get('id')
                if q_id:
                    processed_ids.add(q_id)
                    is_correct = data.get('correct', False)

                    # Update overall stats
                    if is_correct:
                        correct_count += 1
                    total += 1

                    # Update domain-specific stats
                    domain = get_domain_from_id(q_id)
                    if domain and domain in domain_stats:
                        domain_stats[domain]['total'] += 1
                        if is_correct:
                            domain_stats[domain]['correct'] += 1
            except json.JSONDecodeError:
                continue

# ========== New: ORM Evaluation Template ==========
GENERAL_ORM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]
"""

ORM_USER_TEMPLATE = """
Problem: {problem}
Answer 1: {answer_1}
Answer 2: {answer_2}
"""
# --- 7. Evaluate Answers and Write JSONL Line-by-Line ---

def evaluate_one(answer_dict, client, GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE):
    q_id = answer_dict.get('id')
    if not q_id:
        return None, f"Warning: Question '{answer_dict.get('question', '')}' is missing an 'id' field, skipping."
    question = answer_dict['question']
    gold_value = answer_dict['answer']  # Directly use the 'answer' field as the gold standard
    generated_response = answer_dict.get('response', '')  # LLM output
    extracted_answer = extract_answer(generated_response)

    # Construct the ORM evaluation prompt
    system_prompt = GENERAL_ORM_PROMPT
    user_prompt = ORM_USER_TEMPLATE.format(problem=question, answer_1=extracted_answer, answer_2=gold_value)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            temperature=0.0,
            n=1
        )
        result_content = response.choices[0].message.content
        print("\n\n")
        print(f"[Evaluation Model Output] {result_content}")
        # Parse [[YES]]/[[NO]]
        if "[[YES]]" in result_content and "[[NO]]" not in result_content:
            is_correct = True
        elif "[[NO]]" in result_content:
            is_correct = False
        else:
            is_correct = False
        reason = result_content.strip()
        result_entry = {
            'id': q_id,
            'question': question,
            'gold_answer': gold_value,
            'model_answer': extracted_answer,
            'correct': is_correct,
            'reason': reason
        }
        return result_entry, None
    except Exception as e:
        result_entry = {
            'id': q_id,
            'question': question,
            'gold_answer': gold_value,
            'model_answer': extracted_answer,
            'correct': False,
            'reason': f"API call error: {str(e)}"
        }
        return result_entry, None

# Multi-threaded concurrent evaluation
lock = threading.Lock()
results = []
errors = []

with ThreadPoolExecutor(max_workers=max_workers) as executor, open(eval_path, 'a', encoding='utf-8') as f_out:
    futures = {}
    for answer_dict in answers:
        q_id = answer_dict.get('id')
        if not q_id or q_id in processed_ids:
            continue
        futures[executor.submit(evaluate_one, answer_dict, client, GENERAL_ORM_PROMPT, ORM_USER_TEMPLATE)] = q_id
    for future in as_completed(futures):
        q_id = futures[future]
        try:
            result_entry, warn = future.result()
            if warn:
                print(warn)
            if result_entry:
                with lock:
                    f_out.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                    f_out.flush()
                    processed_ids.add(q_id)

                    # Update overall stats
                    is_correct = result_entry['correct']
                    if is_correct:
                        correct_count += 1
                    total += 1

                    # Update domain-specific stats
                    domain = get_domain_from_id(q_id)
                    if domain and domain in domain_stats:
                        domain_stats[domain]['total'] += 1
                        if is_correct:
                            domain_stats[domain]['correct'] += 1
        except Exception as exc:
            print(f"An exception occurred while evaluating ID {q_id}: {exc}")

# --- 8. Final Accuracy Report ---
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

if total > 0:
    # Overall accuracy
    overall_accuracy = correct_count / total
    print(f"\n📊 Overall Performance:")
    print(f"   Total: {total} questions")
    print(f"   Correct: {correct_count}")
    print(f"   Accuracy: {overall_accuracy:.2%}")

    # Domain-specific accuracy
    print(f"\n📂 Performance by Domain:")
    for domain in ['financial_zh', 'financial_en', 'education_en', 'real_estate_en']:
        stats = domain_stats[domain]
        if stats['total'] > 0:
            domain_accuracy = stats['correct'] / stats['total']
            print(f"   {domain:20s}: {stats['correct']:3d}/{stats['total']:3d} = {domain_accuracy:6.2%}")
        else:
            print(f"   {domain:20s}: No questions evaluated")

    print("\n" + "="*60)
else:
    print("\n⚠️  No matching questions and answers found.")
    print("="*60)
