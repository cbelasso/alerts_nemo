import ast
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
import re
import time
from typing import List, Literal, Optional

from llm_parallelization.parallelization import (
    NEMO,
    FlexibleSchemaProcessor,
)
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

num_replication = 5
min_comment_length = 50
max_comment_length = 1000
gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]
max_model_len = 2048 * 5
gpu_memory_utilization = 0.95
batch_size = 25
multiplicity = 1
model = NEMO

output_file_name = f"rewrite_prompts_results_nemo_min_chars_{min_comment_length}_max_{max_comment_length}_project_agnostic"
output_folder_path = "/home/clyde/workspace/alerts_detection_llama/scripts/generation/datasets/"

input_dataframe_path = "/home/clyde/workspace/alerts_detection_llama/scripts/generation/datasets/combined_claude_dg_generated.csv"
reference_dataframe_path = "/home/clyde/workspace/alerts_detection_llama/scripts/generation/datasets/all_data_gov_wo_qa.csv"
project_name = "all"


# -----------------------------
# Schema
# -----------------------------
class SyntheticComment(BaseModel):
    rewritten_comment: str = Field(description="The rewritten comment text")


# -----------------------------
# Prompt Builder
# -----------------------------
def get_rewrite_generation_prompt(reference_comment: str, input_comment: str) -> str:
    prompt = f"""You are given two texts:

1. A reference text that defines the target **stylistic attributes** (tone, structure, phrasing, vocabulary, elaboration, grammatical tendencies, punctuation tendencies, whitespaces, indentation, extraneous punctuation, abbreviations, emojis, shoutouts, capitalization, misspelling, redundancies, etc.)
2. An input comment that defines the **content** to preserve.

Your task is to rewrite the input comment to mimic the **style** of the reference text **without adding, inferring, or expanding** any meaning beyond what is explicitly in the input. 
If the input is very short, keep it short — only modify surface features (e.g., capitalization, formality, punctuation) to align with the reference style. 
Never insert new ideas, explanations, actors, or any other superfluous information.

Reference Text:
{reference_comment}

Input Comment:
{input_comment}

Respond only in JSON format:
{{"rewritten_comment": "<your rewritten version>"}}"""
    return prompt


def load_dataframe(file_path: str) -> pd.DataFrame:
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Input file not found at: {file_path}")

    print(f"Loading input data from: {file_path}")
    ext = Path(file_path).suffix.lower()

    if ext in [".csv", ".txt"]:
        return pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif ext in [".pkl", ".pickle"]:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def parse_annotations(annotation_str):
    if pd.isna(annotation_str):
        return []
    try:
        if isinstance(annotation_str, str):
            try:
                return ast.literal_eval(annotation_str)
            except (ValueError, SyntaxError):
                return [annotation_str]
        elif isinstance(annotation_str, list):
            return annotation_str
        else:
            return [str(annotation_str)]
    except Exception as e:
        print(f"Error parsing annotation: {annotation_str}, Error: {e}")
        return []


def preprocess_input_dataframe(
    input_dataframe: pd.DataFrame, num_replication: int = 1
) -> pd.DataFrame:
    df = (
        pd.concat([input_dataframe] * num_replication, ignore_index=True)
        if num_replication > 1
        else input_dataframe.copy()
    )
    rename_columns = {"text": "input_comment", "comment": "input_comment"}
    df.rename(
        columns={k: v for k, v in rename_columns.items() if k in df.columns}, inplace=True
    )
    return df


def preprocess_reference_dataframe(
    reference_dataframe: pd.DataFrame, project_name: str
) -> pd.DataFrame:
    if project_name == "eec":
        project_filter = "2"
    elif project_name == "sce":
        project_filter = "10"
    elif project_name == "all":
        project_filter = "all"
    else:
        raise ValueError("Invalid project name. Choose 'eec' or 'sce'.")

    if project_filter == "all":
        filtered = reference_dataframe.copy()
    else:
        filtered = reference_dataframe[
            reference_dataframe["project_ids"].str.contains(project_filter)
        ].copy()
    filtered.rename(columns={"text": "reference_comment"}, inplace=True)

    filtered = filtered[
        filtered["reference_comment"].str.len().between(min_comment_length, max_comment_length)
    ].copy()

    filtered["parsed_annotations"] = filtered["annotations"].apply(parse_annotations)
    filtered = filtered.explode("parsed_annotations").reset_index(drop=True)
    filtered["original_target_path"] = filtered["parsed_annotations"].apply(
        lambda x: str(x) if x else ""
    )
    filtered["reference_target_path"] = filtered["original_target_path"]
    filtered["target_path_lower"] = filtered["original_target_path"].str.lower()

    def path_to_sorted_elements(path):
        if not path:
            return ""
        elements = [e.strip() for e in path.split(">")]
        return "|".join(sorted(elements))

    filtered["path_elements_sorted"] = filtered["target_path_lower"].apply(
        path_to_sorted_elements
    )
    filtered = (
        filtered[filtered["target_path_lower"] != ""]
        .drop_duplicates(subset="reference_comment")
        .reset_index(drop=True)
    )
    return filtered


# -----------------------------
# Reference Selection
# -----------------------------
def is_too_similar(a: str, b: str, threshold: float = 0.9) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold


def select_reference_comment(
    input_comment: str, reference_df: pd.DataFrame, used_refs: dict, random_state: int
) -> str:
    """Select a stylistically similar reference (±30% length), distinct from input, and unused for this input."""
    input_comment = str(input_comment).strip()
    input_key = input_comment.lower()
    input_len = len(input_comment)
    min_len = max(5, int(input_len * 0.7))
    max_len = int(input_len * 1.3)

    eligible_refs = reference_df[
        reference_df.apply(
            lambda x: (
                min_len <= len(str(x["reference_comment"])) <= max_len
                and not is_too_similar(str(x["reference_comment"]), input_comment)
                and x.name not in used_refs[input_key]
            ),
            axis=1,
        )
    ]

    if eligible_refs.empty:
        eligible_refs = reference_df[~reference_df.index.isin(used_refs[input_key])]
    if eligible_refs.empty:
        eligible_refs = reference_df  # last fallback

    ref_row = eligible_refs.sample(n=1, random_state=random_state)
    ref_comment = ref_row["reference_comment"].values[0]
    ref_idx = ref_row.index[0]
    used_refs[input_key].add(ref_idx)
    return ref_comment


# -----------------------------
# Safe parsing
# -----------------------------
def safe_parse_rewrite_results(results, prompts):
    for i, res in enumerate(results):
        try:
            prompts[i]["rewritten_comment"] = res.rewritten_comment
        except Exception as e:
            print(f"⚠️ Failed to parse rewritten_comment for prompt {i}: {e}")
            prompts[i]["rewritten_comment"] = ""
    return prompts


def build_reference_buckets(reference_df: pd.DataFrame, bin_size: int = 20):
    """
    Group reference comments into buckets based on their length (in characters)
    to enable fast selection of stylistically similar comments.
    """
    reference_df = reference_df.copy()
    reference_df["length_bin"] = (
        reference_df["reference_comment"].str.len() // bin_size
    ) * bin_size
    buckets = defaultdict(list)
    for idx, row in reference_df.iterrows():
        buckets[row["length_bin"]].append((idx, row["reference_comment"]))
    return buckets


def select_reference_comment_fast(
    input_comment: str,
    reference_buckets: dict,
    used_refs: dict,
    random_state: int,
    bin_size: int = 20,
    similarity_threshold: float = 0.9,
    sample_per_bin: int = 30,
):
    """
    Fast reference selection with pre-bucketed candidates.
    - Avoids full DataFrame filtering per input.
    - Enforces ±30% length range.
    - Ensures distinct and unused references.
    """
    input_comment = str(input_comment).strip()
    input_key = input_comment.lower()
    input_len = len(input_comment)
    if input_len == 0:
        return random.choice(random.choice(list(reference_buckets.values())))[1]

    min_len = max(5, int(input_len * 0.7))
    max_len = int(input_len * 1.3)
    min_bin = (min_len // bin_size) * bin_size
    max_bin = (max_len // bin_size) * bin_size

    # Collect eligible candidates from relevant bins
    eligible_candidates = []
    for b in range(min_bin, max_bin + bin_size, bin_size):
        if b in reference_buckets:
            eligible_candidates.extend(reference_buckets[b])

    if not eligible_candidates:
        # fallback: all refs
        eligible_candidates = sum(reference_buckets.values(), [])

    # Filter out used refs
    used = used_refs[input_key]
    eligible_candidates = [(i, c) for i, c in eligible_candidates if i not in used]

    if not eligible_candidates:
        eligible_candidates = sum(reference_buckets.values(), [])

    # Randomly sample a few candidates to test similarity
    rng = random.Random(random_state)
    subset = rng.sample(eligible_candidates, min(sample_per_bin, len(eligible_candidates)))

    for idx, ref in subset:
        if not is_too_similar(ref, input_comment, threshold=similarity_threshold):
            used.add(idx)
            return ref

    # fallback if all are similar
    idx, ref = rng.choice(eligible_candidates)
    used.add(idx)
    return ref


def clean_excel_string(s):
    if isinstance(s, str):
        # Remove illegal control characters for Excel
        s = re.sub(r"[\000-\010]|[\013-\014]|[\016-\037]", "", s)
        # Optional: replace escaped CR/LF artifacts like _x000D_ with actual newlines or spaces
        s = s.replace("_x000D_", " ")
    return s


def main():
    input_df = load_dataframe(input_dataframe_path)
    ref_df = load_dataframe(reference_dataframe_path)
    ref_df = preprocess_reference_dataframe(ref_df, project_name)
    input_df = preprocess_input_dataframe(input_df, num_replication=num_replication)
    input_df.reset_index(drop=True, inplace=True)
    ref_df.reset_index(drop=True, inplace=True)

    # Precompute buckets for fast lookup
    # reference_buckets = build_reference_buckets(ref_df)
    #
    # used_refs = defaultdict(set)
    prompts = []

    for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Building prompts"):
        input_comment = str(row["input_comment"]).strip()
        # Sample a reference comment randomly
        reference_comment = ref_df.sample(n=1, random_state=idx)["reference_comment"].values[0]

        # reference_comment = select_reference_comment_fast(
        #     input_comment,
        #     reference_buckets,
        #     used_refs,
        #     random_state=idx
        # )
        prompt_text = get_rewrite_generation_prompt(reference_comment, input_comment)

        prompt_row = row.to_dict()
        prompt_row.update({"reference_comment": reference_comment, "prompt": prompt_text})
        prompts.append(prompt_row)

    print(f"✅ Built {len(prompts)} prompts for rewrite generation.")

    processor = FlexibleSchemaProcessor(
        llm=model,
        batch_size=batch_size,
        max_model_len=max_model_len,
        gpu_list=gpu_list,
        gpu_memory_utilization=gpu_memory_utilization,
        multiplicity=multiplicity,
    )

    prompt_texts = [p["prompt"] for p in prompts]

    processor.process_with_schema(prompts=prompt_texts, schema=SyntheticComment)
    results: List[SyntheticComment] = processor.parse_results_with_schema(
        schema=SyntheticComment
    )

    prompts = safe_parse_rewrite_results(results, prompts)
    prompts_df = pd.DataFrame(prompts)

    original_cols = list(input_df.columns)
    additional_cols = ["reference_comment", "prompt", "rewritten_comment"]
    prompts_df = prompts_df[original_cols + additional_cols]

    output_file = Path(f"{output_folder_path}/{output_file_name}.pkl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # prompts_df.to_pickle(output_file)
    prompts_df = prompts_df.applymap(clean_excel_string)
    prompts_df.to_csv(f"{output_folder_path}/{output_file_name}.csv", index=False)
    print(f"🎉 Rewritten prompts saved to: {output_file.resolve()}")
    processor.terminate()


if __name__ == "__main__":
    main()
