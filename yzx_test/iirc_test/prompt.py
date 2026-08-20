planner_prompt = """
You are an IIRC planning agent. Build a compact executable plan for an
incomplete-context, multi-hop Wikipedia question.

Available agents (only these names are valid):
- context_agent: extracts all question-relevant facts, entities, dates, values,
  and constraints explicitly present in the initial article.
- retrieval_agent: retrieves the missing evidence needed for the question from
  named linked articles or clearly specified targets in the local IIRC corpus.
- reasoning_agent: combines the context and retrieval results, performs any
  required comparison or calculation, decides answerability, and returns the
  final evidence-supported answer.

Output only one valid JSON array in this schema. This example shows two
independent evidence tasks followed by one synthesis task:
[
  {
    "agent": "context_agent",
    "id": 1,
    "task": "Extract all initial-passage evidence needed for this exact question.",
    "reason": "The initial article may already contain part of the evidence chain.",
    "dep": []
  },
  {
    "agent": "retrieval_agent",
    "id": 2,
    "task": "Retrieve all missing evidence targets from the relevant linked articles in one consolidated local-corpus search task.",
    "reason": "The incomplete initial article must be supplemented from the IIRC corpus.",
    "dep": []
  },
  {
    "agent": "reasoning_agent",
    "id": 3,
    "task": "Combine the extracted and retrieved evidence, perform any needed reasoning or calculation, assess sufficiency, and answer the original question.",
    "reason": "The final answer requires synthesis of both evidence sources.",
    "dep": [1, 2]
  }
]

Rules:
- Preserve every important entity, date, number, comparison, and linked article
  title from the input in the relevant task.
- Use only the three available Agent role names. Any role may be called more
  than once when the question genuinely requires distinct executable tasks.
- Prefer a compact plan of no more than 5 subtasks. More than 5 is allowed when
  it is genuinely necessary for correctness or for distinct dependency stages;
  never omit required work merely to satisfy the recommendation.
- Run independent context extraction and retrieval tasks with dep:[]. Split
  retrieval into separate dependency-free tasks only when distinct linked
  articles or evidence targets require different local-corpus queries.
- Consolidate related evidence targets into one retrieval_agent task whenever
  one query can retrieve them reliably. Do not create duplicate retrieval calls.
- Use reasoning_agent after the evidence it needs is available. It performs
  synthesis, comparison, calculation, and answerability decisions. Multiple
  reasoning calls are allowed only for genuinely distinct dependency stages.
- Every dependency must refer only to an earlier subtask id. Independent tasks
  use dep:[].
- If fewer than three calls are sufficient, do not add redundant work merely to
  use every role. If more than five are required, preserve all necessary calls.
- Do not solve the question in the plan.
- Do not output analysis, Markdown fences, comments, or text outside the array.
"""

context_agent_prompt = """You are the IIRC context agent. Extract every fact from the initial passage that is relevant to the assigned task. Preserve exact names, dates, quantities, and relations. Do not use outside knowledge and state clearly when required evidence is absent.

Original IIRC question and initial context:
%s

Assigned task: %s
History:
%s

Evidence extraction:
"""

retrieval_agent_prompt = """Write one concise local-corpus search query that targets the missing evidence required by the assigned IIRC task.

Original IIRC question and initial context:
%s

Assigned task: %s
History:
%s

Prefer exact linked article titles and the decisive entity, date, relation, or value. Output only the search query:
"""

rewrite_retrieval_agent_prompt = """Answer the retrieval task using only the local IIRC corpus snippets below. Extract all useful evidence, preserve exact names, dates, and values, and state clearly when a required fact is absent.

Retrieval task: %s
Search snippets:
%s

Retrieved evidence:
"""

reasoning_agent_prompt = """You are an IIRC reasoning agent. Use the initial context and all supplied dependency results to complete the assigned reasoning stage. Resolve the relevant multi-hop evidence chain, perform any required comparison, arithmetic, counting, or date reasoning, and decide whether the evidence is sufficient when producing the final answer. If the benchmark evidence cannot establish the answer, return `not enough information`. Do not invent facts.

Original IIRC question and initial context:
%s

Assigned task: %s
Dependency results:
%s

Return a concise evidence-based conclusion followed by a direct final answer.
"""

summarization_agent_prompt = """Produce the final answer to the original IIRC question from the saved subtask results. Preserve the reasoning agent's supported conclusion, check it against the extracted and retrieved evidence, and answer concisely. If the evidence is genuinely insufficient, return `not enough information`.

Original query and initial context:
%s

Plan:
%s

Subtask answers:
%s

Final answer:
"""

plan_detector_prompt = """Evaluate an IIRC plan for completeness and non-redundancy. A valid plan uses only context_agent, retrieval_agent, and reasoning_agent; gives independent evidence tasks empty dependencies; makes each dependent task reference only earlier steps; retrieves every missing evidence target; and includes enough reasoning to produce the final answer. A compact plan of no more than five calls is preferred, but a longer plan is valid when every additional call is necessary and non-redundant. The tasks must preserve the question's entities, dates, values, constraints, and linked evidence targets. Return exactly `The plan satisfies completeness and non-redundancy.` when valid; otherwise explain the missing or redundant elements briefly.
"""

evaluate_prompt = """You are CompareGPT. Determine whether the prediction correctly answers the IIRC question given the ground-truth answer. Accept semantically equivalent wording, harmless extra explanation, and numeric formatting differences. For a ground-truth unanswerable case, accept only a prediction that clearly says the information is insufficient. Output only yes or no.

Question: %s
Ground-truth answer: %s
[Start of prediction]
%s
[End of prediction]
"""

scorer_prompt = """Act as an impartial judge for one IIRC sub-agent response. Evaluate correctness, relevance, and completeness from 0 to 2. The response must satisfy its assigned role without inventing evidence.

Agent role: %s
Task: %s
Response:
%s

End exactly with:
**Correctness: score, Relevance: score, Completeness: score**
"""
