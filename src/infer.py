"""
CorpusQA Inference Script

This script performs inference on the CorpusQA benchmark using various LLMs.
It supports asynchronous processing for efficient batch inference.

Usage:
    python infer.py --prompt_file /path/to/dataset.jsonl --model gemini-2.5-flash

Author: CorpusQA Team
License: MIT
"""

import json
import os
import argparse
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
import time

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Infer with prompts and save the original answer and LLM response separately")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the input JSONL file containing prompts")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="runs")
    return parser.parse_args()

# --- Configuration ---
args = parse_args()
prompt_file_path = args.prompt_file
model = args.model
MAX_CONCURRENCY = args.concurrency
output_dir = args.output_dir

# --- Initialize asynchronous OpenAI client ---
client = AsyncOpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY"), # Get API Key from environment variable, e.g., export DASHSCOPE_API_KEY="your_key"
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def read_prompt_jsonl(path):
    """
    Read a jsonl file containing id, prompt, question, and answer fields
    """
    prompts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    # Read id, prompt, question, and answer simultaneously
                    if 'id' in item and 'prompt' in item and 'question' in item and 'answer' in item:
                        prompts.append({
                            'id': item['id'],
                            'prompt': item['prompt'],
                            'question': item['question'],
                            'answer': item['answer'] # New: Read the answer field
                        })
                    else:
                        print(f"Warning: Skipping line missing id, prompt, question, or answer field: {line.strip()}")
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse prompt file: {str(e)}")
    return prompts

def save_result(output_file, q_id, question, original_answer, llm_response):
    """
    Save the result.
    - "question": From the input file
    - "answer": From the input file
    - "response": The response from the LLM
    """
    result = {
        "id": q_id,
        "question": question,
        "answer": original_answer,  # The answer read directly from the input file
        "response": llm_response,    # The actual response from the LLM
        "model": model
    }
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()

async def process_prompt_async(item, semaphore):
    """
    Asynchronously process a single prompt, call the API, and return a dictionary
    containing the original information and the LLM's reply.
    """
    async with semaphore:
        q_id = item['id']
        prompt = item['prompt']
        question = item['question']
        original_answer = item['answer'] # Get the original answer

        try:
            # prompt is a list for a multi-turn conversation, each message needs a 'type' field
            messages = []
            for msg in prompt:
                m = msg.copy()
                if 'type' not in m:
                    m['type'] = 'text'
                messages.append(m)

            params = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            if not model.startswith("o3"):
                params["temperature"] = 0.0

            response = await client.chat.completions.create(**params)
            llm_response = response.choices[0].message.content.strip()

            # Include all necessary information in the return result
            return {
                "q_id": q_id,
                "question": question,
                "original_answer": original_answer,
                "response": llm_response
            }
        except Exception as e:
            print(f"  [Error] Processing prompt {q_id} failed: {str(e)}")
            return None

async def main():
    prompts = read_prompt_jsonl(prompt_file_path)
    if not prompts:
        print("No valid prompts were read, exiting.")
        return

    # Generate the output filename, including the model name and the input jsonl filename
    input_file_base = os.path.splitext(os.path.basename(prompt_file_path))[0]
    output_file = os.path.join(output_dir, f"{model}_{input_file_base}.jsonl")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    processed_qids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_qids.add(json.loads(line).get('id'))
                    except json.JSONDecodeError as e:
                        print(f"Warning: JSON parsing failed for file {output_file}: {e}")

    prompts_to_process = [q for q in prompts if q['id'] not in processed_qids]

    if not prompts_to_process:
        print(f"All prompts have already been processed, skipping.")
        return

    print(f"There are {len(prompts_to_process)} new prompts to process...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [process_prompt_async(item, semaphore) for item in prompts_to_process]
    saved_count = 0

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing Prompts"):
        try:
            result = await future
            if result:
                save_result(
                    output_file,
                    result["q_id"],
                    result["question"],
                    result["original_answer"], # Pass the original answer
                    result["response"]         # Pass the LLM response
                )
                saved_count += 1
        except Exception as e:
            print(f"  [Error] A task failed during execution: {e}")

    print(f"Processing complete. Successfully saved {saved_count} new results.")

if __name__ == "__main__":
    asyncio.run(main())
