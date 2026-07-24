import argparse
import json

from get_response import normalize_plan, plan_execution
from planner import Planner, response_text
from utils import is_valid_json


def build_plan_with_llm(query, model):
    planner = Planner(model=model)
    response = planner.plan(query)
    content = response_text(response)
    if not is_valid_json(content):
        raise ValueError(f"Planner did not return valid JSON:\n{content}")
    return json.loads(content)


def run_query(query, plan=None, planner_fn=None, model="gpt-4o"):
    if plan is None:
        plan = planner_fn(query) if planner_fn else build_plan_with_llm(query, model)
    plan = normalize_plan(plan)
    _, subtasks_response, final_answer = plan_execution(query, plan, model=model)
    return {
        "query": query,
        "plan": plan,
        "subtasks_response": subtasks_response,
        "final_answer": final_answer,
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal AOP query -> planner -> agents runner.")
    parser.add_argument("query", help="User query to solve.")
    parser.add_argument("--model", default="gpt-4o", help="Model used for planner and agents.")
    parser.add_argument("--plan-json", help="Optional JSON plan string from another planner, e.g. a diffusion model.")
    parser.add_argument("--plan-file", help="Optional path to a JSON plan file.")
    args = parser.parse_args()

    plan = None
    if args.plan_json:
        plan = json.loads(args.plan_json)
    elif args.plan_file:
        with open(args.plan_file, "r", encoding="utf-8") as f:
            plan = json.load(f)

    result = run_query(args.query, plan=plan, model=args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
