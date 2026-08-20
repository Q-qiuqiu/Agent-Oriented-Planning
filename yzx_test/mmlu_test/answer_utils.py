import re
from collections import Counter


CHOICES = "ABCDEFGHIJ"


def extract_answer_choice(text):
    if not isinstance(text, str) or not text.strip():
        return None

    patterns = [
        r"final\s+answer\s*(?:is|:)?\s*\(?([A-J])\)?",
        r"answer\s*(?:is|:)?\s*\(?([A-J])\)?",
        r"option\s*\(?([A-J])\)?",
        r"\(([A-J])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    stripped = text.strip().upper()
    if stripped in CHOICES:
        return stripped
    return None


def accuracy_summary(rows, prediction_field="predicted_answer"):
    valid = [row for row in rows if row.get("answer") in CHOICES]
    correct = [
        row
        for row in valid
        if row.get(prediction_field) == row.get("answer")
    ]
    by_category = {}
    categories = sorted({row.get("category") for row in valid})
    for category in categories:
        category_rows = [row for row in valid if row.get("category") == category]
        category_correct = sum(
            row.get(prediction_field) == row.get("answer")
            for row in category_rows
        )
        by_category[category] = {
            "count": len(category_rows),
            "correct": category_correct,
            "accuracy": category_correct / len(category_rows) if category_rows else 0.0,
        }

    predictions = Counter(row.get(prediction_field) for row in valid)
    return {
        "count": len(valid),
        "correct": len(correct),
        "accuracy": len(correct) / len(valid) if valid else 0.0,
        "parse_failure_count": sum(
            row.get(prediction_field) not in CHOICES for row in valid
        ),
        "prediction_distribution": dict(sorted(predictions.items(), key=lambda item: str(item[0]))),
        "by_category": by_category,
    }
