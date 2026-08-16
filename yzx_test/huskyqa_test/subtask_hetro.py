import argparse
import json
import time
from pathlib import Path

from evaluate_agent_fit import (
    build_prompt,
    format_for_scorer,
    parse_scores,
    summarize,
)
from openai_compat import run_chat_completion
from prompt import scorer_prompt


# Only edit these values when switching heterogeneous model combinations.
# Assignment order: code_agent, math_agent, search_agent, commonsense_agent.
# 1b aliases: l=llama, g=gemma, q=qwen3, h=hunyuan, f=lfm, m=minicpm,
#             d=deepseek, qm=qwen-math, qc=qwen-coder, i=internlm, s=smollm.
# 3b aliases: l=llama, g=gemma, q=qwen3, p=phi4, m=minicpm.
MODEL_SIZE = "1b"
AGENT_ASSIGNMENT = "qc_q_d_m"
PLAN_VARIANT = "llada_now"

AGENT_ORDER = (
    "code_agent",
    "math_agent",
    "search_agent",
    "commonsense_agent",
)

# Aliases are scoped by MODEL_SIZE, so the same short name can represent the
# corresponding model family in the 1B and 3B pools.
MODEL_PRESETS = {
    "1b": {
        "l": (
            "/data/labshare/Param/llama/llama3/Llama-3.2-1B-Instruct",
            "http://10.137.144.97:7021/v1",
        ),
        "g": (
            "/data/labshare/Param/gemma-3-1b-it",
            "http://10.137.144.97:7022/v1",
        ),
        "q": (
            "/data/labshare/Param/Qwen/Qwen3-1.7B",
            "http://10.137.144.97:7023/v1",
        ),
        "h": (
            "/data/labshare/Param/Hunyuan-1.8B-Instruct",
            "http://10.137.144.97:7024/v1",
        ),
        "f": (
            "/data/labshare/Param/LFM2.5-1.2B-Instruct",
            "http://10.137.144.97:7025/v1",
        ),
        "m": (
            "/data/labshare/Param/MiniCPM5-1B",
            "http://10.137.144.97:7026/v1",
        ),
        "d": (
            "/data/labshare/Param/DeepSeek-R1-Distill-Qwen-1.5B",
            "http://10.137.144.97:7027/v1",
        ),
        "qm": (
            "/data/labshare/Param/Qwen/Qwen2.5-Math-1.5B-Instruct",
            "http://10.137.144.97:7028/v1",
        ),
        "qc": (
            "/data/labshare/Param/Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "http://10.137.144.97:7029/v1",
        ),
        "i": (
            "/data/labshare/Param/internlm2_5-1_8b-chat",
            "http://10.137.144.97:7030/v1",
        ),
        "s": (
            "/data/labshare/Param/SmolLM2-1.7B-Instruct",
            "http://10.137.144.97:7031/v1",
        ),
    },
    "3b": {
        "l": (
            "/data/labshare/Param/llama/llama3/Llama-3.2-3B-Instruct",
            "http://10.137.144.97:7011/v1",
        ),
        "g": (
            "/data/labshare/Param/gemma-3-4b-it",
            "http://10.137.144.97:7012/v1",
        ),
        "q": (
            "/data/labshare/Param/Qwen/Qwen3-4B-Instruct-2507",
            "http://10.137.144.97:7013/v1",
        ),
        "p": (
            "/data/labshare/Param/Phi-4-mini-instruct",
            "http://10.137.144.97:7014/v1",
        ),
        "m": (
            "/data/labshare/Param/MiniCPM3-4B",
            "http://10.137.144.97:7015/v1",
        ),
    },
}


# Initial mapping based on the cache files currently under huskyqa_test/cache.
# Verify shared family caches when adding results from a new model variant.
SEARCH_CACHE_BY_MODEL = {
    "/data/labshare/Param/llama/llama3/Llama-3.2-3B-Instruct": "cache/search_cache_llama3.json",
    "/data/labshare/Param/llama/llama3/Llama-3.2-1B-Instruct": "cache/search_cache_llama3.json",
    "/data/labshare/Param/gemma-3-4b-it": "cache/search_cache_gemma.json",
    "/data/labshare/Param/gemma-3-1b-it": "cache/search_cache_gemma.json",
    "/data/labshare/Param/Qwen/Qwen3-4B-Instruct-2507": "cache/search_cache_qwen.json",
    "/data/labshare/Param/Qwen/Qwen3-1.7B": "cache/search_cache_qwen.json",
    "/data/labshare/Param/Qwen/Qwen2.5-Math-1.5B-Instruct": "cache/search_cache_qwen.json",
    "/data/labshare/Param/Qwen/Qwen2.5-Coder-1.5B-Instruct": "cache/search_cache_qwen.json",
    "/data/labshare/Param/Phi-4-mini-instruct": "cache/search_cache_phi4.json",
    "/data/labshare/Param/MiniCPM3-4B": "cache/search_cache_minicpm.json",
    "/data/labshare/Param/MiniCPM5-1B": "cache/search_cache_minicpm.json",
    "/data/labshare/Param/Hunyuan-1.8B-Instruct": "cache/search_cache_hunyuan.json",
    "/data/labshare/Param/LFM2.5-1.2B-Instruct": "cache/search_cache_lfm.json",
    "/data/labshare/Param/DeepSeek-R1-Distill-Qwen-1.5B": "cache/search_cache_deepseek.json",
    "/data/labshare/Param/internlm2_5-1_8b-chat": "cache/search_cache_internlm.json",
    "/data/labshare/Param/SmolLM2-1.7B-Instruct": "cache/search_cache_smollm.json",
}


def build_agent_config(model_size, assignment):
    try:
        model_pool = MODEL_PRESETS[model_size]
    except KeyError as exc:
        valid_sizes = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(
            f"Unknown MODEL_SIZE {model_size!r}; expected one of: {valid_sizes}"
        ) from exc

    aliases = assignment.split("_")
    if len(aliases) != len(AGENT_ORDER):
        raise ValueError(
            f"AGENT_ASSIGNMENT {assignment!r} must contain exactly four aliases "
            "in code_math_search_commonsense order"
        )

    unknown_aliases = sorted(set(aliases) - set(model_pool))
    if unknown_aliases:
        valid_aliases = ", ".join(sorted(model_pool))
        raise ValueError(
            f"Unknown {model_size} model aliases {unknown_aliases}; "
            f"expected aliases from: {valid_aliases}"
        )

    config = {}
    for agent_name, alias in zip(AGENT_ORDER, aliases):
        model, api_url = model_pool[alias]
        config[agent_name] = {
            "alias": alias,
            "model": model,
            "api_url": api_url,
            "api_key": "empty",
            "temperature": 0.0,
            "timeout": 120,
        }
    return config


AGENT_CONFIG = build_agent_config(MODEL_SIZE, AGENT_ASSIGNMENT)


def search_cache_path_for_agent_config():
    model = AGENT_CONFIG["search_agent"]["model"].rstrip("/")
    try:
        return SEARCH_CACHE_BY_MODEL[model]
    except KeyError as exc:
        known_models = "\n  - ".join(sorted(SEARCH_CACHE_BY_MODEL))
        raise ValueError(
            f"No search cache mapping for search_agent model {model!r}. "
            f"Add it to SEARCH_CACHE_BY_MODEL. Known models:\n  - {known_models}"
        ) from exc


# Other experiment defaults. Model assignments and result names are derived
# from MODEL_SIZE, AGENT_ASSIGNMENT, and PLAN_VARIANT above.
RESULTS_DIR = f"huskyqa_test/results_{MODEL_SIZE}_{PLAN_VARIANT}"
CONFIG = {
    "mode": "respond",
    "plans": f"benchmarks/huskyqa/huskyqa_plans_{PLAN_VARIANT}.json",
    "responses": f"{RESULTS_DIR}/subtask_hetro_responses_{AGENT_ASSIGNMENT}.json",
    "output": f"{RESULTS_DIR}/subtask_hetro_scores_{AGENT_ASSIGNMENT}.json",
    "limit": None,
    "force": False,
    "retry_errors": True,
    "retry_empty_search_results": True,
    "search_backend": "cache_fallback",
    "ddgs_proxy": "http://10.134.110.145:10808",
    "ddgs_timeout": 30,
    "ddgs_retries": 3,
    "ddgs_region": "us-en",
    "ddgs_backends": ["duckduckgo", "brave", "startpage", "bing", "yahoo", "yandex"],
    "ddgs_results_per_backend": 10,
    "ddgs_max_workers": 3,
    "search_top_k": 10,
    "search_cache_path": search_cache_path_for_agent_config(),
    "judge_api_url": "http://10.137.144.97:7001/v1",
    "judge_api_key": "empty",
    "judge_model": "/data/labshare/Param/Qwen/Qwen3-30B-A3B-Instruct-2507",
    #"judge_model": "/home/yzx/models/Qwen3-30B-A3B-Instruct-2507",
    "judge_temperature": 0.0,
    "judge_timeout": 120,
}


def load_json(path, default=None):
    if not path or not Path(path).exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temporary.replace(output)


def normalize_step_id(value):
    return str(value)


def ordered_records(records_by_index, plans):
    order = [record.get("source_index") for record in plans]
    seen = set()
    records = []
    for source_index in order:
        if source_index in records_by_index and source_index not in seen:
            records.append(records_by_index[source_index])
            seen.add(source_index)
    records.extend(record for key, record in records_by_index.items() if key not in seen)
    return records


def answer_for_history(step_record):
    return step_record.get("response") or step_record.get("raw_output") or ""


def build_history(step, completed_by_id):
    dependencies = step.get("dep") or []
    if not dependencies:
        return "None"

    history = []
    missing = []
    failed = []
    for dependency in dependencies:
        dependency_id = normalize_step_id(dependency)
        record = completed_by_id.get(dependency_id)
        if record is None:
            missing.append(dependency)
        elif record.get("error"):
            failed.append(dependency)
        else:
            history.append(f"Subtask {dependency}: {answer_for_history(record)}")

    if missing or failed:
        details = []
        if missing:
            details.append(f"missing dependencies={missing}")
        if failed:
            details.append(f"failed dependencies={failed}")
        raise RuntimeError(", ".join(details))
    return "\n".join(history) or "None"


def validate_agent_config(agent_name):
    if agent_name not in AGENT_CONFIG:
        raise ValueError(f"No AGENT_CONFIG entry for {agent_name!r}")
    config = AGENT_CONFIG[agent_name]
    for key in ["model", "api_url"]:
        if not config.get(key):
            raise ValueError(f"Missing AGENT_CONFIG[{agent_name!r}][{key!r}]")
    return config


def execute_step(plan_record, step, history):
    agent_name = step.get("agent") or step.get("name") or step.get("name_1")
    agent_config = validate_agent_config(agent_name)
    task = step.get("task")
    if not task:
        raise ValueError("Plan step has no task")

    prompt = build_prompt(agent_name, plan_record["query"], task, history)
    raw_output = run_chat_completion(
        agent_config["model"],
        prompt,
        agent_config["api_url"],
        agent_config.get("api_key", ""),
        agent_config.get("timeout", 120),
        agent_config.get("temperature", 0.0),
    )
    scorer_response, metadata = format_for_scorer(
        agent_name,
        raw_output,
        task,
        agent_config["model"],
        agent_config["api_url"],
        agent_config.get("api_key", ""),
        agent_config.get("timeout", 120),
        agent_config.get("temperature", 0.0),
        search_backend=CONFIG["search_backend"],
        ddgs_proxy=CONFIG["ddgs_proxy"],
        ddgs_timeout=CONFIG["ddgs_timeout"],
        ddgs_retries=CONFIG["ddgs_retries"],
        ddgs_region=CONFIG["ddgs_region"],
        ddgs_backends=CONFIG["ddgs_backends"],
        ddgs_results_per_backend=CONFIG["ddgs_results_per_backend"],
        ddgs_max_workers=CONFIG["ddgs_max_workers"],
        search_top_k=CONFIG["search_top_k"],
        search_cache_path=CONFIG["search_cache_path"],
    )
    response = metadata.get("rewritten_response") or scorer_response or raw_output
    return {
        "id": step.get("id"),
        "task": task,
        "agent": agent_name,
        "reason": step.get("reason"),
        "dep": step.get("dep") or [],
        "model": agent_config["model"],
        "api_url": agent_config["api_url"],
        "history": history,
        "raw_output": raw_output,
        "response": response,
        "scorer_response": scorer_response,
        "metadata": metadata,
        "error": None,
    }


def base_response_record(plan_record):
    return {
        "source": plan_record.get("source"),
        "source_index": plan_record.get("source_index"),
        "query": plan_record.get("query"),
        "answer": plan_record.get("answer"),
        "planner_model": plan_record.get("planner_model"),
        "steps": [],
        "error": None,
    }


def has_empty_search_results(step, previous):
    agent_name = previous.get("agent") or step.get("agent")
    snippets = (previous.get("metadata") or {}).get("search_snippets")
    return agent_name == "search_agent" and snippets == []


def execute_plans(
    plans,
    responses_path,
    limit=None,
    force=False,
    retry_errors=True,
    retry_empty_search_results=True,
):
    existing = [] if force else load_json(responses_path, []) or []
    records_by_index = {record.get("source_index"): record for record in existing}
    selected = plans[:limit] if limit else plans

    for plan_record in selected:
        source_index = plan_record.get("source_index")
        response_record = records_by_index.get(source_index) or base_response_record(plan_record)
        records_by_index[source_index] = response_record

        if plan_record.get("error") or not plan_record.get("plan"):
            response_record["error"] = plan_record.get("error") or "planner returned no steps"
            save_json(responses_path, ordered_records(records_by_index, plans))
            continue

        step_records = {
            normalize_step_id(record.get("id")): record
            for record in response_record.get("steps", [])
        }

        for step in plan_record["plan"]:
            step_id = normalize_step_id(step.get("id"))
            previous = step_records.get(step_id)
            retry_empty_search = (
                previous
                and retry_empty_search_results
                and has_empty_search_results(step, previous)
            )
            if (
                previous
                and previous.get("error") is None
                and previous.get("response")
                and not force
                and not retry_empty_search
            ):
                continue
            if previous and previous.get("error") and not retry_errors and not force:
                continue
            if retry_empty_search and not force:
                print(
                    f"retry source={source_index} | step={step.get('id')} "
                    "| reason=empty_search_snippets",
                    flush=True,
                )

            started = time.time()
            try:
                history = build_history(step, step_records)
                step_record = execute_step(plan_record, step, history)
            except Exception as exc:
                agent_name = step.get("agent") or step.get("name") or step.get("name_1")
                agent_config = AGENT_CONFIG.get(agent_name, {})
                step_record = {
                    "id": step.get("id"),
                    "task": step.get("task"),
                    "agent": agent_name,
                    "reason": step.get("reason"),
                    "dep": step.get("dep") or [],
                    "model": agent_config.get("model"),
                    "api_url": agent_config.get("api_url"),
                    "history": None,
                    "raw_output": None,
                    "response": None,
                    "scorer_response": None,
                    "metadata": {},
                    "error": str(exc),
                }
            step_record["time"] = time.time() - started
            step_records[step_id] = step_record
            response_record["steps"] = [
                step_records[normalize_step_id(item.get("id"))]
                for item in plan_record["plan"]
                if normalize_step_id(item.get("id")) in step_records
            ]
            response_record["error"] = (
                "one or more subtask executions failed"
                if any(record.get("error") for record in response_record["steps"])
                else None
            )
            save_json(responses_path, ordered_records(records_by_index, plans))
            print(
                f"respond source={source_index} | step={step.get('id')} | agent={step_record.get('agent')} "
                f"| model={step_record.get('model')} | error={step_record.get('error')}",
                flush=True,
            )

    return ordered_records(records_by_index, plans)


def flatten_steps(response_records):
    rows = []
    for record in response_records:
        for step in record.get("steps", []):
            row = dict(step)
            row.update(
                {
                    "source": record.get("source"),
                    "source_index": record.get("source_index"),
                    "query": record.get("query"),
                    "answer": record.get("answer"),
                    "subtask_id": step.get("id"),
                }
            )
            rows.append(row)
    return rows


def judge_key(row):
    return (
        row.get("source_index"),
        normalize_step_id(row.get("subtask_id")),
        row.get("agent"),
        row.get("model"),
    )


def judge_rows(response_records, output_path, force=False):
    rows = flatten_steps(response_records)
    existing_output = {} if force else load_json(output_path, {}) or {}
    existing_by_key = {judge_key(row): row for row in existing_output.get("rows", [])}
    judged = []

    for row in rows:
        previous = existing_by_key.get(judge_key(row))
        if (
            previous
            and previous.get("judge_output")
            and previous.get("scores")
            and previous.get("scorer_response") == row.get("scorer_response")
        ):
            judged.append(previous)
            continue

        record = dict(row)
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
                prompt = scorer_prompt % (record["agent"], record["task"], record["scorer_response"])
                judge_output = run_chat_completion(
                    CONFIG["judge_model"],
                    prompt,
                    CONFIG["judge_api_url"],
                    CONFIG["judge_api_key"],
                    CONFIG["judge_timeout"],
                    CONFIG["judge_temperature"],
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
        result = {"rows": judged, "summary": summarize(judged)}
        save_json(output_path, result)
        print(
            f"judge source={record.get('source_index')} | step={record.get('subtask_id')} "
            f"| agent={record.get('agent')} | score={record['scores']['total']} "
            f"| error={record.get('judge_error')}",
            flush=True,
        )

    result = {"rows": judged, "summary": summarize(judged)}
    save_json(output_path, result)
    return result


def print_run_config(args):
    print(
        f"Model assignment | size={MODEL_SIZE} | name={AGENT_ASSIGNMENT} "
        f"| order={','.join(AGENT_ORDER)}",
        flush=True,
    )
    for agent_name in AGENT_ORDER:
        agent_config = AGENT_CONFIG[agent_name]
        print(
            f"  {agent_name}: alias={agent_config['alias']} "
            f"| model={agent_config['model']} | api={agent_config['api_url']}",
            flush=True,
        )
    print(f"Plans: {args.plans}", flush=True)
    print(f"Responses: {args.responses}", flush=True)
    print(f"Scores: {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Execute planner-selected subtasks with heterogeneous agent APIs, then judge offline results."
    )
    parser.add_argument("--mode", choices=["respond", "judge", "all"], default=CONFIG["mode"])
    parser.add_argument("--plans", default=CONFIG["plans"])
    parser.add_argument("--responses", default=CONFIG["responses"])
    parser.add_argument("--output", default=CONFIG["output"])
    parser.add_argument("--limit", type=int, default=CONFIG["limit"])
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    args = parser.parse_args()
    print_run_config(args)

    plans = load_json(args.plans, []) or []
    if not plans:
        raise ValueError(f"No planner records found in {args.plans}")

    if args.mode in {"respond", "all"}:
        print(
            "Search cache mapping "
            f"| model={AGENT_CONFIG['search_agent']['model']} "
            f"| path={CONFIG['search_cache_path']}",
            flush=True,
        )
        response_records = execute_plans(
            plans,
            args.responses,
            args.limit,
            args.force,
            CONFIG["retry_errors"],
            CONFIG["retry_empty_search_results"],
        )
        print(f"Saved heterogeneous subtask responses to {args.responses}", flush=True)
    else:
        response_records = load_json(args.responses, []) or []

    if args.mode in {"judge", "all"}:
        if not response_records:
            raise ValueError(f"No response records found in {args.responses}")
        result = judge_rows(response_records, args.output, args.force)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"Saved judged results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
