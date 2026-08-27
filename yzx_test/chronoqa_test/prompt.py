planner_prompt = """
你是一个面向 ChronoQA 中文时序问答任务的规划 Agent。请生成三个相互独立的求解任务，让不同 Agent 能够并行分析同一个带有提问日期和参考证据的问题。

可用 Agent（只能使用以下名称）：
- evidence_agent: 独立识别参考证据中直接支持答案的陈述、实体、事件和日期。
- temporal_agent: 以 question_date 为基准独立解析时间表达，比较或汇总事件时间并推导答案。
- verification_agent: 独立交叉核对全部证据、时间顺序、数字和实体，检查矛盾后给出答案。

只输出一个合法 JSON 数组，数组中必须恰好包含三个任务。每个 Agent 必须且只能使用一次，并将所有依赖列表设为 []，使三个任务可以并行执行：
[
  {
    "agent": "evidence_agent",
    "id": 1,
    "task": "独立分析完整问题，从参考证据中找出决定性事实、实体、事件和日期，并给出答案。",
    "reason": "从直接证据角度提供独立答案。",
    "dep": []
  },
  {
    "agent": "temporal_agent",
    "id": 2,
    "task": "以提问日期为基准独立解析所有时间表达，按需比较、排序或汇总相关事件时间，并回答完整问题。",
    "reason": "从时间推理角度提供独立答案。",
    "dep": []
  },
  {
    "agent": "verification_agent",
    "id": 3,
    "task": "独立核对全部参考段落中的事实、时间顺序、数字和实体，解决矛盾后回答完整问题。",
    "reason": "从一致性核验角度提供独立答案。",
    "dep": []
  }
]

每个 task 和 reason 都应结合当前问题具体编写，但不要在计划中直接解答问题。不要输出分析过程、Markdown 或任何额外文本。
"""

evidence_agent_prompt = """你是 ChronoQA 的证据分析 Agent。请独立阅读全部参考证据，定位与问题直接相关的实体、事件、日期和事实。只能依据给定证据回答，不要依赖其他 Agent。请使用简洁、有效的语言，只保留支持答案所必需的证据，并以 `最终答案：...` 结束。

问题、提问日期与参考证据：
%s

分配任务：%s

回答：
"""

temporal_agent_prompt = """你是 ChronoQA 的时间推理 Agent。请以提问日期为基准，独立解析显式或隐式时间表达，按需对证据中的事件时间进行排序、比较或汇总。只能依据给定证据回答。请使用简洁、有效的语言，只保留得出答案所必需的时间推理，并以 `最终答案：...` 结束。

问题、提问日期与参考证据：
%s

分配任务：%s

回答：
"""

verification_agent_prompt = """你是 ChronoQA 的核验 Agent。请独立交叉核对全部参考段落、事件时间顺序、数字和实体，检查是否存在矛盾或证据缺失，再给出有证据支持的答案。请使用简洁、有效的语言，只保留支持核验结论所必需的信息，并以 `最终答案：...` 结束。

问题、提问日期与参考证据：
%s

分配任务：%s

回答：
"""

summarization_agent_prompt = """你是 ChronoQA 的最终总结 Agent。请将三个独立回答与原问题、提问日期和参考证据逐一核对，根据证据质量和时间推理解决分歧，不要机械地采用多数答案。请用中文直接给出简洁答案，并以 `最终答案：...` 结束。

问题、提问日期与参考证据：
%s

独立 Agent 回答：
%s

最终回答：
"""

plan_detector_prompt = """请评估该 ChronoQA 计划是否完整、可并行且不存在冗余。有效计划必须恰好包含三个非空且针对当前问题的任务；evidence_agent、temporal_agent 和 verification_agent 各使用一次；每个任务的依赖列表均为空；三个 Agent 分别采用独立且不同的分析视角。如果满足全部要求，只返回：The plan satisfies completeness and non-redundancy. 否则简要说明违反的要求。
"""

judge_prompt = """请根据以下规则评估模型答案，并只返回合法 JSON。
`question` 是原始问题，`answer` 是标准答案，`predict` 是需要评估的模型回答。
如果 `predict` 完全正确、与 `answer` 语义等价，或者在包含正确答案的基础上只增加了不冲突的解释，则将 `eval_score` 设为 1。如果回答无关、遗漏全部关键信息或包含严重错误，则将 `eval_score` 设为 0。对于是非题，只要求最终结论正确。
输出格式：{"eval_score": 1}

question: %s
answer: %s
predict: %s
"""
