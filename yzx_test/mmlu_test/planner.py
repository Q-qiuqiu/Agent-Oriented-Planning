import json
import os
from pathlib import Path

import requests

from prompt import planner_prompt


def _load_api_config():
    url = os.environ.get("AOP_API_URL")
    authorization = os.environ.get("AOP_AUTHORIZATION")
    if url and authorization:
        return {"url": url, "Authorization": authorization}

    candidates = [
        Path(__file__).resolve().parent / "keys" / "gptapi_key.json",
        Path(__file__).resolve().parent.parent / "keys" / "gptapi_key.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)

    raise FileNotFoundError(
        "Missing API config. Set AOP_API_URL and AOP_AUTHORIZATION, "
        "or provide keys/gptapi_key.json."
    )


def _chat(messages, model="gpt-4o", temperature=0.0):
    config = _load_api_config()
    headers = {
        "Content-Type": "application/json",
        "Authorization": config["Authorization"],
    }
    payload = {
        "model": model,
        "messages": messages,
        "n": 1,
        "temperature": temperature,
    }
    response = requests.post(config["url"], json=payload, headers=headers, timeout=120)
    response.raise_for_status()
    return response.json()


def response_text(response):
    data = response.get("data", response)
    if "response" in data:
        data = data["response"]
    return data["choices"][0]["message"]["content"]


class Planner:
    def __init__(self, model="gpt-4o", system_prompt=planner_prompt):
        self.model = model
        self.system_prompt = system_prompt

    def plan(self, query):
        return _chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ],
            model=self.model,
        )


def agent(prompt, model="gpt-4o"):
    return _chat([{"role": "user", "content": prompt}], model=model)


# Compatibility with the original repository name.
planner_gpt = Planner
