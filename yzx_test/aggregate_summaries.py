#!/usr/bin/env python3
"""aggregate_summaries.py

聚合指定 bench / 结果目录下某类结果文件中的 "summary" 部分。

结果文件命名规则: <前缀>_<后缀>.json  (例如 subtask_hetro_scores_d_d_d.json)
脚本会遍历目录下所有匹配 <前缀>_*.json 的文件，提取每个文件的 summary，
按后缀整合后输出到屏幕，并附上一份跨后缀加权汇总 (_overall)。

用法:
    # 使用脚本顶部默认配置
    python aggregate_summaries.py

    # 命令行覆盖（优先级高于顶部默认值）
    python aggregate_summaries.py -b chronoqa -r results_1b_base_llama3 -p subtask_hetro_scores
    python aggregate_summaries.py --bench mmlu --results results_1b_full_llada --prefix summary_score
"""

import argparse
import json
import sys
from pathlib import Path

# ==================== 默认配置（可直接修改） ====================
BENCH = "mmlu"                        # bench 名，对应 <BENCH>_test 目录
RESULTS_DIR = "results_1b_full_llada"    # 结果目录名
#FILE_PREFIX = "summary_score"             # 结果文件前缀（不带后缀和 .json）
FILE_PREFIX = "subtask_hetro_scores"             # 结果文件前缀（不带后缀和 .json）


# 只保留 summary 里的哪些字段（任意层级匹配，嵌套结构会保留路径）；
# 可用 (bench, 前缀) 或 bench 做键，(bench, 前缀) 优先；都不在表中的聚合全部字段
SUMMARY_KEYS = {
    ("mmlu", "summary_score"): ["count", "correct", "accuracy", "parse_failure_count"],
    ("mmlu", "subtask_hetro_scores"): ["by_agent", "count", "correct", "accuracy", "parse_failure_count"],
    ("huskyqa", "subtask_hetro_scores"): ["recommended_agents"],
    ("iirc", "subtask_hetro_scores"): ["recommended_agents"],
}

# 输出时后缀的排列顺序；未列出的后缀按字母序排在后面
SUFFIX_ORDER = [
    "q_q_q", "g_g_g", "l_l_l", "m_m_m", "d_d_d",
    "qm_qm_qm", "qc_qc_qc", "i_i_i", "s_s_s",
]
# ===============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# accuracy 类字段不做简单求和，汇总时按 correct / count 重新计算
_RATIO_KEYS = {"accuracy"}


def _filter_keys(node, keys: list):
    """过滤 summary 字段：某一层的键有直接匹配时只保留匹配项（不再深入其他 dict）；
    该层没有匹配时递归进入子 dict 查找，保留匹配路径。"""
    if not isinstance(node, dict):
        return node
    if any(k in keys for k in node):
        return {k: v for k, v in node.items() if k in keys}
    out = {}
    for k, v in node.items():
        if isinstance(v, dict):
            sub = _filter_keys(v, keys)
            if sub:
                out[k] = sub
    return out


def _merge_summary(acc: dict, cur: dict) -> dict:
    """把 cur 的数值累加进 acc（accuracy 等比率字段跳过，最后统一重算）。

    列表等不可累加的值不并入总汇总。
    """
    for k, v in cur.items():
        if k in _RATIO_KEYS or isinstance(v, list):
            continue
        if isinstance(v, dict):
            sub = acc.get(k)
            if not isinstance(sub, dict):
                sub = {}
            _merge_summary(sub, v)
            if sub:  # 子 dict 没有可累加内容时不留下空壳
                acc[k] = sub
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            acc[k] = acc.get(k, 0) + v
        else:
            acc.setdefault(k, v)
    return acc


def _recompute_ratios(node: dict) -> None:
    """递归重算 accuracy = correct / count（退化为 eligible_count）。"""
    if "accuracy" in node or ("correct" in node and ("count" in node or "eligible_count" in node)):
        correct = node.get("correct", 0)
        total = node.get("count") or node.get("eligible_count") or 0
        node["accuracy"] = correct / total if total else 0.0
    for v in node.values():
        if isinstance(v, dict):
            _recompute_ratios(v)


def collect_summaries(results_path: Path, prefix: str, keys: list | None = None) -> dict:
    """遍历 <prefix>_*.json，返回 {后缀: summary}，并附 _overall 汇总。

    keys 不为 None 时，每个 summary 只保留这些字段（任意层级匹配）。
    若所有 summary 都无可累加的数值（如只含列表），则不输出 _overall。
    """
    pattern = f"{prefix}_*.json"
    files = sorted(results_path.glob(pattern))
    if not files:
        print(f"[错误] 在 {results_path} 下没有找到匹配 {pattern} 的文件", file=sys.stderr)
        sys.exit(1)

    merged = {}
    overall: dict = {}
    for fp in files:
        suffix = fp.name[len(prefix) + 1 : -len(".json")]
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[警告] 跳过无法读取的文件 {fp.name}: {e}", file=sys.stderr)
            continue
        summary = data.get("summary")
        if summary is None:
            print(f"[警告] {fp.name} 中没有 summary 字段，已跳过", file=sys.stderr)
            continue
        if keys is not None:
            summary = _filter_keys(summary, keys)
        merged[suffix] = summary
        _merge_summary(overall, summary)

    _recompute_ratios(overall)

    # 按 SUFFIX_ORDER 重排，未列出的后缀按字母序排在后面
    ordered = {k: merged[k] for k in SUFFIX_ORDER if k in merged}
    for k in sorted(merged):
        if k not in ordered:
            ordered[k] = merged[k]
    if overall:
        ordered["_overall"] = overall
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="聚合结果文件中的 summary 部分（按文件后缀整合 + 总汇总）"
    )
    parser.add_argument("-b", "--bench", default=BENCH,
                        help=f"bench 名，对应 <bench>_test 目录（默认: {BENCH}）")
    parser.add_argument("-r", "--results", default=RESULTS_DIR,
                        help=f"结果目录名（默认: {RESULTS_DIR}）")
    parser.add_argument("-p", "--prefix", default=FILE_PREFIX,
                        help=f"结果文件前缀（默认: {FILE_PREFIX}）")
    parser.add_argument("-k", "--keys", default=None,
                        help="只保留 summary 里的这些顶层字段，逗号分隔；"
                             "传 all 表示聚合全部字段（默认: 按顶部 SUMMARY_KEYS 配置）")
    args = parser.parse_args()

    if args.keys is not None:
        keys = None if args.keys.strip().lower() == "all" else [
            k.strip() for k in args.keys.split(",") if k.strip()
        ]
    else:
        keys = SUMMARY_KEYS.get((args.bench, args.prefix), SUMMARY_KEYS.get(args.bench))

    results_path = SCRIPT_DIR / f"{args.bench}_test" / args.results
    if not results_path.is_dir():
        print(f"[错误] 结果目录不存在: {results_path}", file=sys.stderr)
        sys.exit(1)

    print(f"# bench={args.bench}  results={args.results}  prefix={args.prefix}\n")
    merged = collect_summaries(results_path, args.prefix, keys)
    print(json.dumps(merged, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
