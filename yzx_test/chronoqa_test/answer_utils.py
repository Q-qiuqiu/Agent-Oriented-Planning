import json
import re


def extract_final_answer(text):
    if not isinstance(text, str) or not text.strip():
        return None
    patterns = [r"最终答案\s*[：:]\s*(.+)", r"final\s+answer\s*[：:]\s*(.+)"]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    return text.strip()


def extract_eval_score(text):
    if not isinstance(text, str):
        raise ValueError("Judge response is empty")
    cleaned = text.replace("```json", "").replace("```", "").strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        score = value.get("eval_score") if isinstance(value, dict) else None
        if score in (0, 1, 0.0, 1.0):
            return int(score)
    raise ValueError(f"No binary eval_score found in judge response: {cleaned}")


def accuracy_summary(rows, score_field="eval_score"):
    valid = [row for row in rows if row.get(score_field) in (0, 1)]
    score = sum(int(row[score_field]) for row in valid)
    return {
        "count": len(rows),
        "eligible_count": len(valid),
        "correct": score,
        "accuracy": score / len(valid) if valid else 0.0,
        "judge_failure_count": len(rows) - len(valid),
    }
