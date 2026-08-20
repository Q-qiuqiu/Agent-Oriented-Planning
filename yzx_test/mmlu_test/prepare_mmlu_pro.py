import json
import os
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset


# MMLU-Pro has 14 categories. Sampling 25 per category gives 350 questions,
# satisfying the requested 300-1000 total range.
CONFIG = {
    "dataset": "TIGER-Lab/MMLU-Pro",
    "split": "test",
    "per_category": 25,
    "seed": 20260820,
    "output": "benchmarks/mmlu_pro/mmlu_pro_sampled.json",
}

CHOICES = "ABCDEFGHIJ"


def format_query(question, options, category):
    option_lines = [
        f"{CHOICES[index]}. {option}"
        for index, option in enumerate(options)
    ]
    return (
        f"Category: {category}\n"
        f"Question: {question}\n"
        "Options:\n"
        + "\n".join(option_lines)
        + "\nSelect the single best answer from A to J."
    )


def save_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary, output)


def sample_dataset(dataset, per_category, seed):
    rows_by_category = {}
    for row in dataset:
        rows_by_category.setdefault(row["category"], []).append(row)

    rng = random.Random(seed)
    selected = []
    for category in sorted(rows_by_category):
        rows = rows_by_category[category]
        if len(rows) < per_category:
            raise ValueError(
                f"Category {category!r} has only {len(rows)} rows; "
                f"cannot sample {per_category}"
            )
        category_rows = rng.sample(rows, per_category)
        category_rows.sort(key=lambda row: int(row["question_id"]))
        for row in category_rows:
            options = list(row["options"])
            if not 2 <= len(options) <= len(CHOICES):
                raise ValueError(
                    f"question_id={row['question_id']} has unsupported "
                    f"option count {len(options)}"
                )
            answer_index = int(row["answer_index"])
            answer = str(row["answer"]).strip().upper()
            if answer != CHOICES[answer_index]:
                raise ValueError(
                    f"Inconsistent answer at question_id={row['question_id']}"
                )
            selected.append(
                {
                    "source": CONFIG["dataset"],
                    "source_index": int(row["question_id"]),
                    "question_id": int(row["question_id"]),
                    "category": row["category"],
                    "src": row["src"],
                    "question": row["question"],
                    "options": options,
                    "answer": answer,
                    "answer_index": answer_index,
                    "query": format_query(row["question"], options, row["category"]),
                }
            )

    # Preserve deterministic category blocks while assigning a stable row index.
    for sample_index, row in enumerate(selected):
        row["sample_index"] = sample_index
    return selected


def main():
    dataset = load_dataset(CONFIG["dataset"], split=CONFIG["split"])
    selected = sample_dataset(
        dataset,
        CONFIG["per_category"],
        CONFIG["seed"],
    )
    save_json(CONFIG["output"], selected)
    category_counts = Counter(row["category"] for row in selected)
    print(f"Saved {len(selected)} questions to {CONFIG['output']}")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()
