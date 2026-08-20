import json


def is_valid_json(value):
    try:
        json.loads(value)
    except (TypeError, ValueError):
        return False
    return True


def simplify_answer(answer, convert_to_str=False):
    if answer is None or answer == "":
        return "[FAIL]"
    if convert_to_str:
        return str(answer)
    return answer
