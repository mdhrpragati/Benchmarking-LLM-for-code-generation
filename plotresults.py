# =========================
# Required Imports
# =========================
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_all_evaluation_plots(
    judge_metrics_file="evaluation_metrics_qwenjudge.xlsx",
    model_metrics_file="evaluation_all_models.xlsx",
    base_output_dir="evaluation_outputs",
    humaneval_name="humaneval",
    humanevalnext_name="humanevalnext",
    dpi=300
):
    """
    Reads evaluation Excel files and generates all plots,
    saving them into structured output directories.
    """

    # =========================
    # Helper Functions
    # =========================
    def ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    def sanitize_filename(name):
        return re.sub(r"[\\/:\*\?\"<>\| ]+", "_", name)

    def annotate_bars(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )

    # =========================
    # Output Directories
    # =========================
    judge_output_dir = os.path.join(base_output_dir, "judge_based_plots")
    model_output_dir = os.path.join(base_output_dir, "model_comparison_plots")
    ensure_dir(judge_output_dir)
    ensure_dir(model_output_dir)

    # =====================================================
    # PART 1: Judge-Based Metrics (HumanEval vs Next)
    # =====================================================
    df = pd.read_excel(judge_metrics_file)
    df["dataset"] = df["dataset"].str.lower()

    agg_df = (
        df.groupby(["judge_model", "test_model", "dataset"])
        .agg(
            mean_avg_score=("avg_score", "mean"),
            mean_diversity=("diversity", "mean"),
            pct_max_score=("max_score", lambda x: (x == 1).mean() * 100),
            pct_min_score=("min_score", lambda x: (x == 0).mean() * 100),
        )
        .reset_index()
    )

    metrics = [
        ("mean_avg_score", "Mean of Average Scores"),
        ("mean_diversity", "Mean of Diversity Scores"),
        ("pct_max_score", "Percentage of Max Scores"),
        ("pct_min_score", "Percentage of Min Scores"),
    ]

    for judge in agg_df["judge_model"].unique():
        judge_df = agg_df[
            (agg_df["judge_model"] == judge) &
            (agg_df["test_model"].str.lower() != "vanilla")
        ]

        test_models = sorted(judge_df["test_model"].unique())
        if not test_models:
            continue

        x = np.arange(len(test_models))
        width = 0.35
        safe_judge = sanitize_filename(judge)

        for metric_col, metric_title in metrics:
            humaneval_vals = []
            humanevalnext_vals = []

            for tm in test_models:
                he = judge_df.query(
                    "test_model == @tm and dataset == @humaneval_name"
                )[metric_col]
                hen = judge_df.query(
                    "test_model == @tm and dataset == @humanevalnext_name"
                )[metric_col]

                humaneval_vals.append(he.values[0] if not he.empty else 0)
                humanevalnext_vals.append(hen.values[0] if not hen.empty else 0)

            fig, ax = plt.subplots(figsize=(10, 6))
            bars1 = ax.bar(x - width / 2, humaneval_vals, width, label="HumanEval")
            bars2 = ax.bar(x + width / 2, humanevalnext_vals, width, label="HumanEvalNext")

            annotate_bars(bars1, ax)
            annotate_bars(bars2, ax)

            ax.set_xticks(x)
            ax.set_xticklabels(test_models, rotation=45, ha="right")
            ax.set_ylabel(metric_title)
            ax.set_xlabel("Test Model")
            ax.set_title(f"{metric_title} (Judge Model: {judge})")
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.6)

            plt.tight_layout()
            plt.savefig(
                os.path.join(judge_output_dir, f"{safe_judge}_{metric_col}.png"),
                dpi=dpi,
                bbox_inches="tight"
            )
            plt.close()

    # =====================================================
    # PART 2: Model Comparison Metrics
    # =====================================================
    df = pd.read_excel(model_metrics_file)

    model_map = {
        "bigcode/starcoder2-7b": "bigcode/starcoder2-7b",
        "deepseek-ai/deepseek-coder-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "qwen/qwen2.5-coder-7b-instruct": "qwen/qwen2.5-coder-7b-instruct",
        "codellama/codellama-7b-python-hf": "codellama/CodeLlama-7b-Python-hf",
        "codellama/codellama-7b-instruct-hf": "codellama/CodeLlama-7b-Instruct-hf",
        "ise-uiuc/magicoder-s-ds-6.7b": "ise-uiuc/Magicoder-S-DS-6.7B",
        "aixcoder/aixcoder-7b": "aiXcoder/aiXcoder-7B",
        "mistralai/mistral-7b-v0.3": "mistralai/Mistral-7B-v0.3",
        "qwen/qwen2.5-3b": "Qwen/Qwen2.5-3B",
    }

    df["model_clean"] = df["model"].str.lower().str.strip().replace(model_map)
    df = df[~df["model_clean"].str.contains("vanillaovo")]

    x_order = [
        "aiXcoder/aiXcoder-7B",
        "bigcode/starcoder2-7b",
        "codellama/CodeLlama-7b-Instruct-hf",
        "codellama/CodeLlama-7b-Python-hf",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "ise-uiuc/Magicoder-S-DS-6.7B",
        "qwen/qwen2.5-coder-7b-instruct",
        "qwen/qwen2.5-3B",
        "mistralai/Mistral-7B-v0.3"
    ]

    def plot_metric(metric_name):
        metric_df = df[df["evaluation_metric"] == metric_name]
        pivot = metric_df.pivot(
            index="model_clean",
            columns="dataset",
            values="result"
        ).reindex(x_order).fillna(0)

        x = np.arange(len(pivot.index))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 6))
        bars1 = ax.bar(x - width / 2, pivot[humaneval_name], width, label="HumanEval")
        bars2 = ax.bar(x + width / 2, pivot[humanevalnext_name], width, label="HumanEvalNext")

        annotate_bars(bars1, ax)
        annotate_bars(bars2, ax)

        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("Models")
        ax.set_title(f"{metric_name} Comparison")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(
            os.path.join(model_output_dir, f"{metric_name}.png"),
            dpi=dpi,
            bbox_inches="tight"
        )
        plt.close()

    for metric in ["pass@1", "pass@5", "avg_unique_solution_rate", "bleu"]:
        plot_metric(metric)


# =========================
# Script Entry Point
# =========================
if __name__ == "__main__":
    generate_all_evaluation_plots()
