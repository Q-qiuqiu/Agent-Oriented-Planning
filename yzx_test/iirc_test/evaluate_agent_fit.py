import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from get_response import _extract_python_code
from iirc_retrieval import search_iirc
from openai_compat import run_chat_completion
from prompt import (
    code_agent_prompt,
    commonsense_agent_prompt,
    math_agent_prompt,
    rewrite_code_agent_prompt,
    rewrite_math_agent_prompt,
    rewrite_search_agent_prompt,
    scorer_prompt,
    search_agent_prompt,
)
from utils import simplify_answer


AGENTS = ["code_agent", "math_agent", "search_agent", "commonsense_agent"]

#/data/labshare/Param/llama/llama3/Llama-3.2-3B-Instruct
#/data/labshare/Param/gemma-3-4b-it
CONFIG = {
    "mode": "respond",
    "models": "/data/labshare/Param/Qwen/Qwen3-4B-Instruct-2507",
    "agents": ["search_agent"],
    "benchmarks": "benchmarks/iirc/iirc_subtask_llama3.json",
    "responses": "iirc_test/results/agent_fit_responses_qwen3.json",
    "output": "iirc_test/results/agent_fit_results_qwen3.json",
    "local_api_url": "http://10.137.144.97:7003/v1",
    "local_api_key": "empty",
    "local_temperature": 0.0,
    "timeout": 120,
    "search_backend": "iirc_sqlite",
    "search_top_k": 5,
    "iirc_sqlite_path": "benchmarks/iirc/context_articles.sqlite3",
    "force": False,
    "judge_api_url": "http://10.137.144.97:7001/v1",
    "judge_api_key": "empty",
    "judge_model": "/data/labshare/Param/Qwen/Qwen3-30B-A3B-Instruct-2507",
    "judge_temperature": 0.0,
    "judge_timeout": 120,
}

DEFAULT_BENCHMARKS = [
    {
        "agent": "math_agent",
        "query": "A store sold 36 notebooks on Monday and 48 on Tuesday. If each notebook costs 7 dollars, what was the total revenue?",
        "task": "Calculate the total revenue from selling 36 notebooks and 48 notebooks at 7 dollars each.",
        "history": "None",
    },
    {
        "agent": "code_agent",
        "query": "Find the median of the numbers 18, 7, 22, 5, 13, 9, 31.",
        "task": "Write Python code to compute the median of [18, 7, 22, 5, 13, 9, 31].",
        "history": "None",
    },
    {
        "agent": "search_agent",
        "query": "In what country is the University of Geneva?",
        "task": "Retrieve the University of Geneva article and identify its country.",
        "history": "None",
        "agent_context": (
            "Question: In what country is the University of Geneva?\n\n"
            "Initial passage: The person attended seminars at the University of Geneva.\n\n"
            "Available linked articles:\n- University of Geneva"
        ),
    },
    {
        "agent": "commonsense_agent",
        "query": "If a glass cup is dropped on a concrete floor, what is likely to happen?",
        "task": "Answer what is likely to happen when a glass cup is dropped on a concrete floor.",
        "history": "None",
    },
]


def load_json(path, default=None):
    if not path:
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_models(models):
    if models is None:
        return None
    if isinstance(models, str):
        return [models]
    return list(models)


def build_prompt(agent_name, query, task, history, agent_context=None):
    original_input = agent_context or query
    if agent_name == "math_agent":
        return math_agent_prompt % (original_input, task, history)
    if agent_name == "code_agent":
        return code_agent_prompt % (original_input, task, history)
    if agent_name == "search_agent":
        return search_agent_prompt % (original_input, task, history)
    if agent_name == "commonsense_agent":
        return commonsense_agent_prompt % (original_input, task, history)
    raise ValueError(f"Unknown agent: {agent_name}")


def iirc_search_snippets(search_query, sqlite_path, top_k):
    results = search_iirc(sqlite_path, search_query, top_k)
    snippets = []
    for item in results:
        title = str(item.get("title") or "").strip()
        body = str(item.get("body") or "").strip()
        text = f"{title}: {body}".strip(": ")
        if text:
            snippets.append(text)
    print(
        f"search {search_query!r} | backend=iirc_sqlite | results={len(snippets)}",
        flush=True,
    )
    return snippets


def search_snippets(
    search_query,
    search_backend,
    search_top_k,
    iirc_sqlite_path,
):
    if search_backend != "iirc_sqlite":
        raise ValueError(
            f"Unsupported IIRC search backend: {search_backend}. Use iirc_sqlite."
        )
    return iirc_search_snippets(
        search_query,
        iirc_sqlite_path,
        search_top_k,
    )


def format_for_scorer(
    agent_name,
    raw_output,
    task,
    model,
    local_api_url,
    local_api_key,
    local_timeout,
    local_temperature,
    search_backend="iirc_sqlite",
    search_top_k=5,
    iirc_sqlite_path="benchmarks/iirc/context_articles.sqlite3",
):
    if agent_name == "math_agent":
        rewrite_prompt = rewrite_math_agent_prompt % (task, raw_output)
        try:
            rewritten = run_chat_completion(
                model,
                rewrite_prompt,
                local_api_url,
                local_api_key,
                local_timeout,
                local_temperature,
            )
        except Exception:
            rewritten = raw_output
        scorer_response = """
[The Start of the Original Response to Solve the Question]
%s
[The End of the Original Response to Solve the Question]
[The Start of the Rewritten Response]
%s
[The End of the Rewritten Response]
""" % (raw_output, rewritten)
        return scorer_response, {"rewritten_response": rewritten}

    if agent_name == "search_agent":
        search_query = raw_output.strip()
        snippets = search_snippets(
            search_query,
            search_backend,
            search_top_k,
            iirc_sqlite_path,
        )
        rewrite_prompt = rewrite_search_agent_prompt % (task, snippets)
        final_answer = run_chat_completion(
            model,
            rewrite_prompt,
            local_api_url,
            local_api_key,
            local_timeout,
            local_temperature,
        )
        return final_answer, {"search_query": search_query, "search_snippets": snippets}

    if agent_name != "code_agent":
        return raw_output, {}

    code = _extract_python_code(raw_output)
    metadata = {"code": code}
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        code_output = simplify_answer(result.stdout.strip(), convert_to_str=True)
        if result.stderr.strip():
            code_output = f"[ERROR] {result.stderr.strip()}"
    except Exception as exc:
        code_output = f"[ERROR] {exc}"

    metadata["code_output"] = code_output
    rewrite_prompt = rewrite_code_agent_prompt % (task, code, code_output)
    try:
        rewritten = run_chat_completion(
            model,
            rewrite_prompt,
            local_api_url,
            local_api_key,
            local_timeout,
            local_temperature,
        )
    except Exception:
        rewritten = code_output

    scorer_response = """
[The Start of the Code to Solve the Question]
%s
[The End of the Code to Solve the Question]
[The Start of the Rewritten Response by Running the Code]
%s
[The End of the Rewritten Response by Running the Code]
""" % (code, rewritten)
    metadata["rewritten_response"] = rewritten
    return scorer_response, metadata


def response_key(model, item):
    return (
        model,
        item.get("agent"),
        item.get("source_index"),
        item.get("subtask_id"),
        item.get("task"),
    )


def generate_responses(
    models,
    benchmarks,
    local_api_url,
    local_api_key,
    local_temperature,
    local_timeout,
    search_backend="iirc_sqlite",
    search_top_k=5,
    iirc_sqlite_path="benchmarks/iirc/context_articles.sqlite3",
    existing_rows=None,
    force=False,
):
    rows = list(existing_rows or [])
    positions = {
        response_key(row.get("model"), row): index
        for index, row in enumerate(rows)
    }
    done = set() if force else {
        (
            row.get("model"),
            row.get("agent"),
            row.get("source_index"),
            row.get("subtask_id"),
            row.get("task"),
        )
        for row in rows
        if row.get("raw_output") is not None or row.get("error")
    }

    for model in models:
        for item in benchmarks:
            key = response_key(model, item)
            if key in done:
                continue
            agent_name = item["agent"]
            query = item["query"]
            agent_context = item.get("agent_context")
            task = item["task"]
            history = item.get("history", "None")
            started = time.time()
            record = {
                "model": model,
                "agent": agent_name,
                "query": query,
                "task": task,
                "history": history,
                "source": item.get("source"),
                "source_index": item.get("source_index"),
                "subtask_id": item.get("subtask_id"),
                "planner_agent": item.get("planner_agent"),
                "answer": item.get("answer"),
                "agent_context": agent_context,
                "answer_type": item.get("answer_type"),
                "original_answer": item.get("original_answer"),
                "article_pid": item.get("article_pid"),
                "article_title": item.get("article_title"),
                "available_links": item.get("available_links") or [],
                "gold_question_links": item.get("gold_question_links") or [],
                "gold_context": item.get("gold_context") or [],
            }
            raw_output = None
            try:
                prompt = build_prompt(
                    agent_name,
                    query,
                    task,
                    history,
                    agent_context,
                )
                raw_output = run_chat_completion(
                    model,
                    prompt,
                    local_api_url,
                    local_api_key,
                    local_timeout,
                    local_temperature,
                )
                scorer_response, metadata = format_for_scorer(
                    agent_name,
                    raw_output,
                    task,
                    model,
                    local_api_url,
                    local_api_key,
                    local_timeout,
                    local_temperature,
                    search_backend,
                    search_top_k,
                    iirc_sqlite_path,
                )
                record.update(
                    {
                        "raw_output": raw_output,
                        "scorer_response": scorer_response,
                        "metadata": metadata,
                        "error": None,
                        "response_time": time.time() - started,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "raw_output": raw_output,
                        "scorer_response": None,
                        "metadata": {},
                        "error": str(exc),
                        "response_time": time.time() - started,
                    }
                )
            if key in positions:
                rows[positions[key]] = record
            else:
                positions[key] = len(rows)
                rows.append(record)
            print(f"respond {model} | {agent_name} | error={record['error']}", flush=True)
    return rows


def parse_scores(text):
    pattern = r"Correctness:\s*([0-2]).*?Relevance:\s*([0-2]).*?Completeness:\s*([0-2])"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if not match:
        return {"correctness": None, "relevance": None, "completeness": None, "total": None}
    correctness, relevance, completeness = [int(value) for value in match.groups()]
    return {
        "correctness": correctness,
        "relevance": relevance,
        "completeness": completeness,
        "total": correctness + relevance + completeness,
    }


def judge_response(row, judge_model, judge_api_url, judge_api_key, judge_timeout, judge_temperature):
    prompt = scorer_prompt % (row["agent"], row["task"], row["scorer_response"])
    return run_chat_completion(
        judge_model,
        prompt,
        judge_api_url,
        judge_api_key,
        judge_timeout,
        judge_temperature,
    )


def judge_responses(rows, judge_model, judge_api_url, judge_api_key, judge_timeout, judge_temperature):
    judged = []
    for row in rows:
        record = dict(row)
        if record.get("judge_output") and record.get("scores"):
            judged.append(record)
            continue
        started = time.time()
        if record.get("error") or not record.get("scorer_response"):
            record.update(
                {
                    "judge_output": None,
                    "scores": {"correctness": 0, "relevance": 0, "completeness": 0, "total": 0},
                    "judge_error": record.get("error") or "missing scorer_response",
                    "judge_time": 0,
                }
            )
        else:
            try:
                judge_output = judge_response(
                    record,
                    judge_model,
                    judge_api_url,
                    judge_api_key,
                    judge_timeout,
                    judge_temperature,
                )
                record.update(
                    {
                        "judge_output": judge_output,
                        "scores": parse_scores(judge_output),
                        "judge_error": None,
                        "judge_time": time.time() - started,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "judge_output": None,
                        "scores": {"correctness": 0, "relevance": 0, "completeness": 0, "total": 0},
                        "judge_error": str(exc),
                        "judge_time": time.time() - started,
                    }
                )
        judged.append(record)
        print(
            f"judge {record.get('model')} | {record.get('agent')} | score={record['scores']['total']} | error={record.get('judge_error')}",
            flush=True,
        )
    return judged


def summarize(rows):
    summary = {}
    for row in rows:
        model = row["model"]
        agent_name = row["agent"]
        summary.setdefault(model, {}).setdefault(
            agent_name,
            {"count": 0, "total": 0, "correctness": 0, "relevance": 0, "completeness": 0},
        )
        scores = row.get("scores") or {}
        bucket = summary[model][agent_name]
        bucket["count"] += 1
        for key in ["total", "correctness", "relevance", "completeness"]:
            bucket[key] += scores.get(key) or 0

    for model, agent_scores in summary.items():
        ranked_source = []
        for agent_name, values in agent_scores.items():
            count = values["count"]
            values["avg_total"] = values["total"] / count if count else 0
            values["avg_correctness"] = values["correctness"] / count if count else 0
            values["avg_relevance"] = values["relevance"] / count if count else 0
            values["avg_completeness"] = values["completeness"] / count if count else 0
            ranked_source.append((agent_name, values))
        ranked = sorted(ranked_source, key=lambda item: item[1]["avg_total"], reverse=True)
        summary[model]["recommended_agents"] = [
            {"agent": agent_name, "avg_total": values["avg_total"]} for agent_name, values in ranked
        ]
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate which agent roles local models are suitable for, with offline response saving."
    )
    parser.add_argument("--mode", choices=["respond", "judge", "all"], default=CONFIG["mode"])
    parser.add_argument("--models", nargs="+", default=CONFIG["models"], help="Local model names served by an OpenAI-compatible API.")
    parser.add_argument("--agents", nargs="+", choices=AGENTS, default=CONFIG["agents"])
    parser.add_argument("--benchmarks", default=CONFIG["benchmarks"], help="Benchmark JSON file.")
    parser.add_argument("--responses", default=CONFIG["responses"], help="Offline response JSON path.")
    parser.add_argument("--output", default=CONFIG["output"], help="Judged output JSON path.")
    parser.add_argument("--local-api-url", default=CONFIG["local_api_url"])
    parser.add_argument("--local-api-key", default=CONFIG["local_api_key"])
    parser.add_argument("--local-temperature", type=float, default=CONFIG["local_temperature"])
    parser.add_argument("--timeout", type=int, default=CONFIG["timeout"], help="Timeout per local model call.")
    parser.add_argument(
        "--search-backend",
        choices=["iirc_sqlite"],
        default=CONFIG["search_backend"],
    )
    parser.add_argument("--search-top-k", type=int, default=CONFIG["search_top_k"])
    parser.add_argument("--iirc-sqlite-path", default=CONFIG["iirc_sqlite_path"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"], help="Regenerate local responses even if they already exist.")
    parser.add_argument("--judge-api-url", default=CONFIG["judge_api_url"])
    parser.add_argument("--judge-api-key", default=CONFIG["judge_api_key"])
    parser.add_argument("--judge-model", default=CONFIG["judge_model"])
    parser.add_argument("--judge-temperature", type=float, default=CONFIG["judge_temperature"])
    parser.add_argument("--judge-timeout", type=int, default=CONFIG["judge_timeout"])
    args = parser.parse_args()

    if args.mode in {"respond", "all"}:
        args.models = normalize_models(args.models)
        if not args.models:
            raise ValueError("--models is required for respond/all mode.")
        benchmarks = load_json(args.benchmarks, DEFAULT_BENCHMARKS)
        if args.agents:
            benchmarks = [item for item in benchmarks if item.get("agent") in args.agents]
        existing = load_json(args.responses, []) if Path(args.responses).exists() else []
        response_rows = generate_responses(
            args.models,
            benchmarks,
            args.local_api_url,
            args.local_api_key,
            args.local_temperature,
            args.timeout,
            args.search_backend,
            args.search_top_k,
            args.iirc_sqlite_path,
            existing,
            args.force,
        )
        save_json(args.responses, response_rows)
        print(f"Saved responses to {args.responses} ({len(response_rows)} rows)")
    else:
        response_rows = load_json(args.responses, [])

    if args.mode in {"judge", "all"}:
        if not args.judge_api_url:
            raise ValueError("Missing judge API URL. Pass --judge-api-url or set JUDGE_OPENAI_API_URL.")
        judge_input = response_rows
        if args.agents:
            judge_input = [row for row in judge_input if row.get("agent") in args.agents]
        selected_keys = {response_key(row.get("model"), row) for row in judge_input}
        existing_result = load_json(args.output, {}) if Path(args.output).exists() else {}
        retained_rows = [
            row
            for row in (existing_result.get("rows", []) if isinstance(existing_result, dict) else [])
            if response_key(row.get("model"), row) not in selected_keys
        ]
        newly_judged_rows = judge_responses(
            judge_input,
            args.judge_model,
            args.judge_api_url,
            args.judge_api_key,
            args.judge_timeout,
            args.judge_temperature,
        )
        judged_rows = retained_rows + newly_judged_rows
        result = {"rows": judged_rows, "summary": summarize(judged_rows)}
        save_json(args.output, result)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Saved judged results to {args.output}")


if __name__ == "__main__":
    main()
