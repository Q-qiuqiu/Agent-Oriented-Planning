planner_prompt = """
You are a planning agent. Decompose the user query into executable subtasks and
choose one suitable agent for each subtask.

Available agents:
- code_agent: writes and runs Python code for precise computations.
- math_agent: solves math questions by step-by-step reasoning.
- search_agent: retrieves missing evidence from the local IIRC Wikipedia corpus.
- commonsense_agent: performs reading comprehension, evidence synthesis, and general reasoning.

Output only valid JSON in this format:
[
  {
    "id": 1,
    "task": "subtask description",
    "agent": "math_agent",
    "reason": "why this agent is suitable",
    "dep": []
  }
]

Rules:
- Include all important entities, numbers, constraints, and dates from the query.
- Use dependencies in "dep" when a subtask needs previous results.
- Use the initial passage and available linked article titles in the user input.
- Add retrieval subtasks when the initial passage does not contain enough information.
- Do not assume that every IIRC question is answerable.
- If the query can be solved by one agent, output one subtask.
"""

math_agent_prompt = """You are a math agent. Solve the subtask using the original query and previous results.

Original query: %s
Subtask: %s
History:
%s

Answer the subtask clearly and concisely.
"""

code_agent_prompt = """You are a code agent. Write Python code to solve the subtask using the original query and previous results.
Return only one Python code block.

Original query: %s
Subtask: %s
History:
%s

Code:
"""

rewrite_code_agent_prompt = """Given the subtask, Python code, and code output, write a concise natural-language answer.

Subtask: %s
Code:
%s
Code output:
%s

Answer:
"""

rewrite_math_agent_prompt = """Given the subtask and the math agent's original answer, rewrite the result into a concise final answer.

Subtask: %s
Original answer:
%s

Answer:
"""

search_agent_prompt = """Write a concise local-corpus search query for the subtask.

Original IIRC question and initial context:
%s

Subtask: %s
History:
%s

Prefer exact linked article titles shown in the initial context when relevant.
Output only the search query:
"""

rewrite_search_agent_prompt = """Answer the search question using the search snippets. Be concise and cite no unavailable details.

Question: %s
Search snippets:
%s

Answer:
"""

commonsense_agent_prompt = """You are a reading-comprehension and commonsense agent. Answer the subtask using the original IIRC question, initial passage, and previous results.

Original query: %s
Subtask: %s
History:
%s

Answer:
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
