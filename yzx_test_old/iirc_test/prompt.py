planner_prompt = """
You are an IIRC planning agent. Decompose the question and its incomplete
Wikipedia context into executable evidence-gathering and reasoning subtasks.

Available agents:
- context_agent: extracts relevant facts, entities, dates, and constraints that
  are explicitly present in the initial passage.
- retrieval_agent: retrieves one missing fact from one named linked article or
  one clearly specified target in the local IIRC Wikipedia corpus.
- reasoning_agent: compares and combines already extracted evidence without
  performing new retrieval or numerical computation.
- calculation_agent: performs arithmetic, counting, date, or other precise
  calculations using already collected evidence.
- answerability_agent: determines whether the collected evidence is sufficient
  to answer the question and identifies a genuinely unanswerable question.

Output only valid JSON in this format:
[
  {
    "id": 1,
    "task": "detailed executable subtask",
    "agent": "context_agent",
    "reason": "why this agent is suitable",
    "dep": []
  }
]

Rules:
- The "agent" value must exactly match one of these five strings:
  "context_agent", "retrieval_agent", "reasoning_agent",
  "calculation_agent", or "answerability_agent". Never shorten, reword, or
  derive variants such as "retrieve_agent" or "reasonability_agent".
- Preserve all important entities, numbers, constraints, dates, and linked
  article titles from the input.
- First identify the independent evidence targets needed to answer the question.
- When the initial passage contains relevant evidence, create one
  context_agent task to extract it.
- Create a separate retrieval_agent task for each independently retrievable
  missing fact. Each retrieval task must name one evidence target and preferably
  one available linked article. Do not combine unrelated facts or articles into
  one retrieval task.
- The context_agent and independently executable retrieval_agent tasks can all
  read the original input, so give them "dep": []. Do not make retrieval depend
  on context extraction when its target is already explicit in the input.
- A task that needs a previous result must list the exact step ids in "dep".
  reasoning_agent, calculation_agent, and answerability_agent tasks normally
  depend on the evidence-gathering steps they consume.
- Use calculation_agent only when an explicit computation is required. Use
  answerability_agent only when evidence sufficiency must be decided.
- Do not invent evidence targets, duplicate a retrieval, or create a subtask
  solely to increase the number of agents.
- If the question genuinely needs only one operation, output one subtask.
- Keep the plan concise, normally no more than five steps.
"""

context_agent_prompt = """You are an IIRC context agent. Extract only the facts requested by the subtask from the initial passage. Do not use outside knowledge or invent missing evidence.

Original IIRC question and initial context:
%s
Subtask: %s
History:
%s

Return the relevant passage evidence and a concise conclusion.
"""

retrieval_agent_prompt = """Write one concise local-corpus search query for the specified missing evidence target.

Original IIRC question and initial context:
%s
Subtask: %s
History:
%s

Prefer the exact linked article title named in the subtask or initial context.
Search for only this subtask's evidence target.
Output only the search query:
"""

rewrite_retrieval_agent_prompt = """Answer the retrieval subtask using only the local-corpus snippets.

Subtask: %s
Search snippets:
%s

Return the requested evidence and state clearly when the snippets do not contain it.
"""

reasoning_agent_prompt = """You are an IIRC reasoning agent. Compare and combine the supplied evidence to solve the subtask. Do not claim facts absent from the initial context or dependency results.

Original IIRC question and initial context:
%s
Subtask: %s
Dependency results:
%s

Return a concise evidence-based conclusion.
"""

calculation_agent_prompt = """You are an IIRC calculation agent. Perform the exact arithmetic, counting, date, or comparison operation requested by the subtask.

Original IIRC question and initial context:
%s
Subtask: %s
Evidence and previous results:
%s

Show the essential calculation and return the result.
"""

rewrite_calculation_agent_prompt = """Rewrite the calculation response as a concise evidence-based answer to the subtask.

Subtask: %s
Calculation response:
%s

Answer:
"""

answerability_agent_prompt = """You are an IIRC answerability agent. Decide whether the available evidence is sufficient to answer the subtask. Treat the question as unanswerable only when the required fact cannot be established from the initial passage and retrieved evidence.

Original IIRC question and initial context:
%s
Subtask: %s
Evidence and previous results:
%s

Return either a supported answer or a concise explanation that the question is unanswerable.
"""

summarization_agent_prompt = """Use the subtask answers to produce the final answer to the original query.

Original query: %s
Plan:
%s
Subtask answers:
%s

Final answer:
"""

plan_detector_prompt = """You are a plan detector responsible for analyzing the completeness and redundancy of the plan. Given the query and the plan formulated to solve the query, which involves several sub-tasks, you should do the following things:
1. **Detect whether the plan satisfies the completeness.**: Evaluate whether the set of subtasks covers all key aspects of the original task including important numbers and nouns. Specifically, check if each important element and requirement from the original task is addressed by at least one subtask. Provide a brief explanation if any key information is missing.
2. **Detect whether the plan satisfies the non-redundancy.**: Evaluate whether any two sub-tasks contain identical information and requirements. If there is any redundant part, list and provide suggestions for optimizing the plan.
---
For example:
Task: If a plane can carry 300 passengers and flies from Brazil to Nigeria with a full load, then returns with only 75% capacity filled, how many passengers in total has it transported between the two countries in one round trip?
Subtask 1: Determine the number of passengers transported from Brazil to Nigeria in one flight with a full load.    Dependency: []
Subtask 2: Determine the number of passengers transported from Nigeria to Brazil in one flight with 75% capacity filled.    Dependency: []
Subtask 3: Calculate the total number of passengers transported between Brazil and Nigeria in one round trip.    Dependency: [1, 2]
Analyse: This plan does not satisfy completeness because the subtask loses the information of 'a plane can carry 300 passengers' of the original task. This plan satisfies non-redundancy because each subtask has a unique focus and there is no overlap in the information covered.
Suggestions: Add the information of 'a plane can carry 300 passengers' to subtask 1 and subtask 2.
---
If there is no need to modify the plan, just return 'The plan satisfies completeness and non-redundancy.'.
"""

evaluate_prompt = """You are CompareGPT, a machine to verify the correctness of predictions. Answer with only yes/no.
You are given a question, the corresponding ground-truth answer and a prediction from a model. Compare the "Ground-truth answer" and the "Prediction" to determine whether the prediction correctly answers the question. The prediction may contain extra information, but a correct prediction includes the ground-truth answer. You can answer "yes" if the prediction includes the ground-truth answer. You must answer "no" if there are any specific details in the ground-truth answer that are not mentioned in the prediction. If the prediction states something as a possibility, treat it as a definitive answer. Note that the error within three decimal places is negligible.
---
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
