import argparse
import json
from pathlib import Path

from answer_utils import accuracy_summary, extract_answer_choice


MODEL_SIZE = "1b"
AGENT_ASSIGNMENT = "q_qm_m"
PLAN_VARIANT = "llada"
RESULTS_DIR = f"mmlu_test/results_{MODEL_SIZE}_{PLAN_VARIANT}"
CONFIG = {
    "input": f"{RESULTS_DIR}/summary_result_{AGENT_ASSIGNMENT}.json",
    "output": f"{RESULTS_DIR}/summary_evaluate_{AGENT_ASSIGNMENT}.json",
}


def main():
    parser = argparse.ArgumentParser(
        description="Score MMLU-Pro final answers by official option exact match."
    )
    parser.add_argument("--input", default=CONFIG["input"])
    parser.add_argument("--output", default=CONFIG["output"])
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as file:
        rows = json.load(file)
    scored = []
    for row in rows:
        item = dict(row)
        prediction = item.get("predicted_answer")
        if not prediction and item.get("final_answer"):
            prediction = extract_answer_choice(item["final_answer"])
        item["predicted_answer"] = prediction
        item["correct"] = prediction == item.get("answer")
        scored.append(item)

    result = {"rows": scored, "summary": accuracy_summary(scored)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    temporary.replace(output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved exact-match evaluation to {args.output}")


if __name__ == "__main__":
    main()
