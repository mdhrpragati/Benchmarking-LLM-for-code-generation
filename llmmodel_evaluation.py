import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from evaluate import load
from transformers import AutoModelForCausalLM, AutoTokenizer

# DEVICE SETUP
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch version:", torch.__version__)
print("Using device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

torch.set_grad_enabled(False)

# HELPER FUNCTIONS
def unique_solution_rate(samples):
    return len(set(samples)) / max(len(samples), 1)

def extract_code(text, prompt):
    if "def " in text:
        return text[text.find("def "):]
    return text[len(prompt):]

# DATASETS AND METRICS
print("Loading datasets and metrics...")
human_eval = load_dataset("openai_humaneval")['test']
human_evalnext = load_dataset("AISE-TUDelft/HumanEvalNext")['train']

datasets_to_load = [
    ("humaneval", human_eval),
    ("humanevalnext", human_evalnext)
]

code_eval_metric = load("code_eval")
bleu_metric = load("bleu")
print("Datasets and metrics loaded!")

# MODELS TO TEST
models_to_test = [
    # Strong general code models 
    "bigcode/starcoder2-7b",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",

    # CodeLlama family 
    "codellama/CodeLlama-7b-Python-hf",
    "codellama/CodeLlama-7b-Instruct-hf",

    #  Instruction-tuned research models 
    "ise-uiuc/Magicoder-S-DS-6.7B",
    "aiXcoder/aiXcoder-7B",
    "mistralai/Mistral-7B-v0.3",
    "Qwen/Qwen2.5-3B",
]

# GENERATION SETTINGS
num_samples_per_problem = 5
max_new_tokens = 512
temperature = 0.7
top_p = 1.0
do_sample = True

# OUTPUT FILE
output_path = "evaluation_all_models.xlsx"

# MAIN LOOP
for model_name in models_to_test:
    print(f"\n=== Loading model: {model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        )
    except RuntimeError as e:
        print(f"FP16 load failed, falling back to 8-bit: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True,
        )

    model.eval()
    model.to(device)

    for dataset_name, dataset in datasets_to_load:
        print(f"\n--- Dataset: {dataset_name} ---")
        test_cases = []
        all_candidates = []
        diversity_scores = []

        for problem in tqdm(dataset, desc="Problems", unit="problem"):
            prompt = problem["prompt"]
            reference = problem.get("test") or problem.get("canonical_solution")
            test_cases.append(reference)

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    num_return_sequences=num_samples_per_problem,
                    pad_token_id=tokenizer.eos_token_id,
                )

            problem_candidates = [extract_code(tokenizer.decode(o, skip_special_tokens=True), prompt) for o in outputs]
            all_candidates.append(problem_candidates)
            diversity_scores.append(unique_solution_rate(problem_candidates))

        # Evaluate metrics
        k_values = [1, 5]
        pass_at_k, _ = code_eval_metric.compute(
            references=test_cases,
            predictions=all_candidates,
            k=k_values,
            num_workers=1,
            timeout=20.0,
        )

        first_preds = [c[0] for c in all_candidates]
        bleu_score = bleu_metric.compute(predictions=first_preds, references=[[r] for r in test_cases])["bleu"]

        results_dict = {
            "pass@1": pass_at_k.get("pass@1", 0.0),
            "pass@5": pass_at_k.get("pass@5", 0.0),
            "avg_unique_solution_rate": float(np.mean(diversity_scores)),
            "bleu": bleu_score,
        }

        rows = [{"evaluation_metric": metric, "model": model_name, "dataset": dataset_name, "result": value} 
                for metric, value in results_dict.items()]
        new_df = pd.DataFrame(rows)

        if os.path.exists(output_path):
            existing_df = pd.read_excel(output_path)
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        final_df.to_excel(output_path, index=False)
        print(f"Results for {model_name} | {dataset_name} appended to {output_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

# CLEAR HUGGINGFACE CACHE
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
if os.path.exists(hf_home):
    import shutil
    shutil.rmtree(hf_home, ignore_errors=True)
    print(f"Hugging Face cache cleared at {hf_home}")

print("Evaluation complete for all models.")