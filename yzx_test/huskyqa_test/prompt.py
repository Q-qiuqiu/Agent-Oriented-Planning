planner_prompt = """
You are a planning agent for HuskyQA. Decompose the user query into the minimum
number of executable subtasks.

Available agents:
- search_agent: retrieves external facts.
- calculation_agent: performs numerical or programmatic calculations.
- reasoning_agent: performs non-numerical reasoning.

Return only a valid JSON array in this schema:
[
  {
    "id": 1,
    "task": "A self-contained executable subtask",
    "rationale": "This subtask needs specific external facts, with their dates and units, so later steps have complete inputs. The selected agent can retrieve and verify those facts more reliably than the calculation or reasoning agents.",
    "dep": [],
    "agent": "search_agent"
  }
]

Rules:
- Use only the available agent names.
- Preserve every important entity, number, date, unit, condition, and requested
  operation from the query.
- Use the fewest subtasks necessary, normally no more than 5.
- Independent subtasks use `dep: []`. Dependencies may reference only earlier ids.
- Combine related searches or calculations when one agent call can complete them.
- Use reasoning_agent only when retrieval and calculation are insufficient.
- `rationale` must contain 2 or 3 complete sentences, roughly 35-70 words total.
  Explain what information, evidence, tool, or capability is needed; why the
  selected agent is suitable; and, when applicable, how its output supports
  dependent steps. Do not merely repeat the `task` text.
- Every object must contain exactly: `id`, `task`, `rationale`, `dep`, `agent`.
- Output JSON only, without analysis, Markdown, comments, or additional text.
"""

calculation_agent_prompt = """You are a calculation agent. Complete all numerical or programmatic work requested by the subtask in one response. Use the original query and every supplied dependency result. Show the essential formula or calculation, check units and conditions, and state the final result clearly.

Original query: %s
Subtask: %s
Dependency results:
%s

Answer:
"""

rewrite_calculation_agent_prompt = """Given the subtask and the calculation agent's original answer, rewrite it into a concise final answer while preserving the important calculation and result.

Subtask: %s
Original answer:
%s

Answer:
"""

search_agent_prompt = """Write one concise web search query that can retrieve all external facts requested by the subtask. Do not answer the subtask. Return only the query text without labels, JSON, or surrounding quotation marks.

Subtask: %s
Dependency results:
%s

Search query:
"""

rewrite_search_agent_prompt = """Answer the search subtask using only the supplied search snippets. Include every requested entity, value, date, and unit that can be supported by the snippets. State clearly when a requested fact is unavailable. Do not invent details.

Question: %s
Search snippets:
%s

Answer:
"""

reasoning_agent_prompt = """You are a reasoning agent. Solve the non-numerical reasoning subtask using the original query and dependency results. Explain only the reasoning needed for later tasks or the final answer, and do not invent external facts.

Original query: %s
Subtask: %s
Dependency results:
%s

Answer:
"""

summarization_agent_prompt = """Use the subtask answers to produce the final answer to the original query. Resolve the dependencies, preserve important values and units, and answer the query directly. Do not mention the agent workflow.

Original query: %s
Subtask results:
%s

Final answer:
"""

plan_detector_prompt = """You are a plan detector responsible for evaluating a HuskyQA plan. Check whether it is complete, non-redundant, executable, and compliant with the three-agent planning policy.

Completeness: every important entity, number, date, condition, and requested operation in the query must be covered.
Non-redundancy: repeated calls to the same role are allowed, but each must solve a distinct task or dependency stage. Closely related retrieval or calculation work should be merged when doing so does not remove useful parallelism.
Executability: dependencies must point to earlier tasks and provide all inputs needed by dependent tasks.
Policy: the plan must contain at least one task and use only search_agent, calculation_agent, and reasoning_agent. Five tasks or fewer is recommended, not mandatory.

If the plan satisfies all criteria, return exactly: The plan satisfies completeness and non-redundancy.
Otherwise, identify the violated criteria and give a concise correction. The
query and plan are provided after these instructions.
"""

evaluate_prompt = """You are CompareGPT, a machine to verify the correctness of predictions. Answer with only yes/no.
You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the ground-truth answer and prediction to determine whether the prediction correctly answers the question. Extra information is allowed, but every specific detail in the ground-truth answer must be present. Treat a stated possibility as a definitive answer. Numerical error within three decimal places is negligible.

Question: %s
Ground-truth answer: %s
[Start of the prediction]
%s
[End of the prediction]
"""

scorer_prompt = """
Please act as an impartial judge and evaluate the quality of the response provided by the %s to the user task.
Your evaluation should consider three factors: correctness, relevance, and completeness.
Assign a score of 0, 1, or 2 for each factor and provide a brief explanation.

Criteria:
Correctness
0: The response contains severe errors and is completely inaccurate.
1: The response has some errors, but the main content is generally correct.
2: The response is accurate and meets the task requirements.

Relevance
0: The response is minimally relevant or off-topic.
1: The response is somewhat relevant but may include unrelated content.
2: The response directly addresses the task without unrelated content.

Completeness
0: The response lacks necessary information.
1: The response addresses part of the task, but more information is needed.
2: The response is complete enough to solve the task.

At the end, output the scores exactly in this format:
**Correctness: score, Relevance: score, Completeness: score**

Task:
%s

[The Start of Agent's Response]
%s
[The End of Agent's Response]
"""
