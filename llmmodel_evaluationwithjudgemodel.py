import os
import re
import ast
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Environment setup
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# Cache cleanup
def clear_hf_cache():
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        print("🧹 Clearing Hugging Face cache...")
        shutil.rmtree(cache_dir, ignore_errors=True)

# Utility functions
def unique_solution_rate(samples):
    return len(set(filter(None, samples))) / max(len(samples), 1)

def extract_single_function(text):
    if text is None:
        return None

    text = text.replace("\x00", "")

    text = re.sub(r"```python|```", "", text)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return None
    func = funcs[0]
    lines = text.splitlines()
    return "\n".join(lines[func.lineno - 1: func.end_lineno])

def clean_code(code):
    if code is None:
        return None
    code = re.sub(r"if __name__ == ['\"]__main__['\"]:.*", "", code, flags=re.DOTALL)
    code = re.sub(r"^\s*import .*", "", code, flags=re.MULTILINE)
    return code.strip()

def is_valid_python(code):
    try:
        compile(code, "<candidate>", "exec")
        return True
    except Exception:
        return False

# Judge function
def judge_code(judge_model, judge_tokenizer, prompt, tests, code):
    eval_prompt = f"""
You are a strict Python code reviewer.

Problem:
{prompt}

Reference tests:
{tests}

Candidate solution:
{code}

Scoring rubric:
- Score 1: Correct for all cases
- Score 0.5: Minor bug or missing edge case
- Score 0: Incorrect or invalid

Respond EXACTLY in this format:
Explanation: <short explanation>
Score: <0 | 0.5 | 1>
"""
    inputs = judge_tokenizer(
        eval_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)

    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    outputs = judge_model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        temperature=0.0,
        pad_token_id=judge_tokenizer.eos_token_id
    )

    text = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)

    score_match = re.search(r"Score:\s*(1(?:\.0)?|0\.5|0(?:\.0)?)", text)
    score = float(score_match.group(1)) if score_match else 0.0

    exp_match = re.search(r"Explanation:\s*(.*)", text, re.DOTALL)
    explanation = exp_match.group(1).strip() if exp_match else text.strip()
    return score, explanation

# Config
num_samples_per_problem = 5

judge_models = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    #"deepseek-ai/deepseek-coder-6.7b-instruct",
    #"mistralai/Mistral-7B-Instruct-v0.3",
]

models_to_test = [
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    #"Qwen/Qwen2.5-Coder-7B-Instruct",
    "bigcode/starcoder2-7b",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-7b-Python-hf",
    #"vanillaOVO/WizardLM-7B-V1.0",
    "ise-uiuc/Magicoder-S-DS-6.7B",
    "aiXcoder/aiXcoder-7B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    #"Qwen/Qwen2.5-3B",
]

metrics_path = "evaluation_metrics_qwenjudge.xlsx"
explanations_path = "evaluation_explanations_qwenjudge.xlsx"

# Load datasets ONCE
datasets = [
    ("HumanEval", load_dataset("openai_humaneval")["test"]),
    ("HumanEvalNext", load_dataset("AISE-TUDelft/HumanEvalNext")["train"]),
]

# Evaluation loop
for test_model_name in models_to_test:

    print(f"\n🚀 Loading test model: {test_model_name}")
    test_tokenizer = AutoTokenizer.from_pretrained(test_model_name, trust_remote_code=True)
    test_tokenizer.pad_token = test_tokenizer.eos_token

    test_model = AutoModelForCausalLM.from_pretrained(
        test_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device).eval()

    for judge_model_name in judge_models:

        # Skip same-family judging
        if "qwen" in judge_model_name.lower() and "qwen" in test_model_name.lower():
            continue
        if "deepseek" in judge_model_name.lower() and "deepseek" in test_model_name.lower():
            continue

        print(f"🧑‍⚖️ Judge: {judge_model_name}")
        judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name, trust_remote_code=True)
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(device).eval()

        batch_results = []
        batch_explanations = []

        for dataset_name, dataset in datasets:
            for idx, problem in tqdm(
                enumerate(dataset),
                total=len(dataset),
                desc=f"{test_model_name} | {judge_model_name} | {dataset_name}"
            ):
                prompt = problem["prompt"]
                tests = problem.get("test", "")

                inputs = test_tokenizer(
                    prompt + "\n\n# Write ONLY the Python function. No explanation.\n",
                    return_tensors="pt"
                )

                inputs.pop("token_type_ids", None)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = test_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    num_return_sequences=num_samples_per_problem,
                    pad_token_id=test_tokenizer.eos_token_id
                )

                candidates = []
                for o in outputs:
                    decoded = test_tokenizer.decode(o, skip_special_tokens=True)
                    candidates.append(clean_code(extract_single_function(decoded)))

                diversity = unique_solution_rate(candidates)
                scores = []

                for c in candidates:
                    if not c or not is_valid_python(c):
                        score, explanation = 0.0, "Syntax error or empty output."
                    else:
                        score, explanation = judge_code(
                            judge_model, judge_tokenizer, prompt, tests, c
                        )

                    scores.append(score)
                    batch_explanations.append({
                        "test_model": test_model_name,
                        "judge_model": judge_model_name,
                        "dataset": dataset_name,
                        "problem_index": idx,
                        "candidate": c,
                        "judge_score": score,
                        "judge_explanation": explanation,
                    })

                batch_results.append({
                    "test_model": test_model_name,
                    "judge_model": judge_model_name,
                    "dataset": dataset_name,
                    "problem_index": idx,
                    "diversity": diversity,
                    "avg_score": np.mean(scores),
                    "max_score": np.max(scores),
                    "min_score": np.min(scores),
                })

        # Append results to Excel
        for path, data in [
            (metrics_path, batch_results),
            (explanations_path, batch_explanations),
        ]:
            df_new = pd.DataFrame(data)
            if os.path.exists(path):
                df_old = pd.read_excel(path)
                df_new = pd.concat([df_old, df_new], ignore_index=True)
            df_new.to_excel(path, index=False)

        # Cleanup judge + cache
        del judge_model, judge_tokenizer
        torch.cuda.empty_cache()
        clear_hf_cache()

    # Cleanup test model + cache
    del test_model, test_tokenizer
    torch.cuda.empty_cache()
    clear_hf_cache()

print("ALL EVALUATIONS COMPLETED SAFELY")
