import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_DIR), str(SCRIPT_DIR.parent)]

from base_llama3_builder import replace_json_array_example, run_builder
from build_subtask_benchmark import (
    AGENTS,
    expand_plans,
    load_queries,
    normalize_plan,
    print_summary,
)
from prompt import planner_prompt


PROMPT_VERSION = "chronoqa_base_llama3_prefetch_then_plan_v4"

OUTPUT_CONTRACT = """对于 base_llama3 对照策略，只输出一个合法 JSON 对象。
对象必须严格按顺序包含两个顶层字段：先输出 `prefetch_agents`，再输出
`plan`。

严格使用以下结构：
{
  "prefetch_agents": [
    {"id": 1, "agent": "evidence_agent"},
    {"id": 2, "agent": "temporal_agent"},
    {"id": 3, "agent": "verification_agent"}
  ],
  "plan": [
    {"agent": "evidence_agent", "id": 1, "task": "...", "reason": "...", "dep": []},
    {"agent": "temporal_agent", "id": 2, "task": "...", "reason": "...", "dep": []},
    {"agent": "verification_agent", "id": 3, "task": "...", "reason": "...", "dep": []}
  ]
}

先在内部确定完整拆分方案，不要输出这个思考过程。随后在任何 task、reason
或 dep 文本之前输出 `prefetch_agents`，按执行顺序列出此时已经识别出的
Agent 调用，并从 1 开始连续编号。

然后输出完整且自包含的 `plan`。后续执行和评分只以 `plan` 为准；其中每个
对象都必须显式包含 `agent`、`id`、`task`、`reason` 和 `dep`。

不要为了让 `plan` 与 `prefetch_agents` 完全一致而删减、重复或扭曲最终计划，
最终 `plan` 的正确性和完整性优先。对于本 benchmark，最终计划必须包含下面
规定的三个独立调用。

并行约束：
- 必须恰好生成三个 task，并按上述顺序将三个 Agent 各使用一次。
- 三个 Agent 都会分别收到完整的原始问题、question_date 和全部参考证据，
  每个 Agent 都必须据此独立分析并独立给出答案。
- 三个 task 的 `dep` 必须逐字输出为 `[]`。禁止输出 `[1]`、`[1, 2]`
  或任何其他依赖。
- temporal_agent 不得依赖 evidence_agent 的输出；它应自己阅读完整证据并
  完成时间推理。
- verification_agent 不得汇总、复查或等待其他 Agent 的输出；它应直接对
  完整原始证据进行独立交叉核验。三个回答将在规划流程之外统一总结。
- 不要输出 JSON 对象之外的介绍、解释、总结或 Markdown。"""

BASE_LLAMA3_PROMPT = replace_json_array_example(
    planner_prompt,
    "只输出一个合法 JSON 数组，数组中必须恰好包含三个任务。",
    OUTPUT_CONTRACT,
)

CONFIG = {
    "input": "benchmarks/chronoqa/chronoqa_sampled.json",
    "plans_output": "benchmarks/chronoqa/chronoqa_plans_base_llama3.json",
    "benchmark_output": "benchmarks/chronoqa/chronoqa_subtask_base_llama3.json",
    "planner_api_url": "http://10.137.144.97:7002/v1",
    "planner_api_key": "empty",
    "planner_model": "/data/labshare/Param/llama/llama3/Meta-Llama-3-8B-Instruct",
    "planner_temperature": 0.0,
    "planner_max_tokens": 1024,
    "timeout": 600,
    "limit": None,
    "source": "czy1999/ChronoQA",
    "prompt_version": PROMPT_VERSION,
    "planner_mode": "prefetch_then_plan",
    "planner_prompt": BASE_LLAMA3_PROMPT,
}


if __name__ == "__main__":
    run_builder(
        config=CONFIG,
        agents=AGENTS,
        load_queries=load_queries,
        normalize_plan=normalize_plan,
        expand_plans=expand_plans,
        print_summary=print_summary,
        description="Build ChronoQA base_llama3 agent-names-first plans.",
    )
