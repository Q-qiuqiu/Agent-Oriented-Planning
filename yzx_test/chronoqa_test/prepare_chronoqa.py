import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


CONFIG = {
    "input": "benchmarks/chronoqa/chronoqa_raw.json",
    "output": "benchmarks/chronoqa/chronoqa_sampled.json",
    "per_temporal_type": 120,
    "seed": 20260826,
}

TEMPORAL_TYPES = ("absolute", "aggregate", "relative")


def format_query(row):
    chunks = "\n".join(
        f"[{index}] {chunk}"
        for index, chunk in enumerate(row["golden_chunks"], start=1)
    )
    return (
        f"提问日期：{row['question_date']}\n"
        f"问题：{row['question']}\n"
        f"参考证据：\n{chunks}\n"
        "请以提问日期为时间基准，仅依据给定的参考证据回答问题。"
    )


def save_json(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary, output)


def sample_dataset(dataset, per_temporal_type, seed):
    rows_by_type = defaultdict(list)
    for source_index, row in enumerate(dataset):
        temporal_type = row.get("temporal_type")
        if temporal_type not in TEMPORAL_TYPES:
            continue
        if not all(row.get(key) for key in ("question", "answer", "question_date", "golden_chunks")):
            continue
        rows_by_type[temporal_type].append((source_index, row))

    rng = random.Random(seed)
    selected = []
    for temporal_type in TEMPORAL_TYPES:
        rows = rows_by_type[temporal_type]
        if len(rows) < per_temporal_type:
            raise ValueError(
                f"Temporal type {temporal_type!r} has only {len(rows)} valid rows; "
                f"cannot sample {per_temporal_type}"
            )
        sampled = rng.sample(rows, per_temporal_type)
        sampled.sort(key=lambda item: item[0])
        for source_index, row in sampled:
            selected.append(
                {
                    "source": "czy1999/ChronoQA",
                    "source_index": source_index,
                    "question_id": f"chronoqa_{source_index}",
                    "question": row["question"],
                    "question_date": row["question_date"],
                    "answer": row["answer"],
                    "temporal_type": temporal_type,
                    "temporal_expression_type": row["temporal_expression_type"],
                    "temporal_scope": row["temporal_scope"],
                    "temporal_granularity": row["temporal_granularity"],
                    "answer_type": row["answer_type"],
                    "reference_document_count": row["reference_document_count"],
                    "golden_chunks": row["golden_chunks"],
                    "golden_chunks_urls": row.get("golden_chunks_urls", []),
                    "query": format_query(row),
                }
            )

    # Preserve deterministic temporal-type blocks and assign stable sample IDs.
    for sample_index, row in enumerate(selected):
        row["sample_index"] = sample_index
    return selected


def main():
    with Path(CONFIG["input"]).open("r", encoding="utf-8") as file:
        dataset = json.load(file)
    selected = sample_dataset(
        dataset,
        CONFIG["per_temporal_type"],
        CONFIG["seed"],
    )
    save_json(CONFIG["output"], selected)
    type_counts = Counter(row["temporal_type"] for row in selected)
    document_counts = Counter(row["reference_document_count"] for row in selected)
    print(f"Saved {len(selected)} questions to {CONFIG['output']}")
    print(f"  temporal_types={dict(type_counts)}")
    print(f"  reference_documents={dict(document_counts)}")


if __name__ == "__main__":
    main()
