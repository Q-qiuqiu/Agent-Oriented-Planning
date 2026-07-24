from pathlib import Path

from iirc_retrieval import search_iirc
from planner import agent, response_text
from prompt import rewrite_search_agent_prompt


# Edit these defaults directly when using the legacy get_response.py entry point.
CONFIG = {
    "sqlite_path": "benchmarks/iirc/context_articles.sqlite3",
    "max_results": 5,
}


class IIRCSearch:
    def __init__(self, sqlite_path=None, max_results=None):
        self.sqlite_path = Path(sqlite_path or CONFIG["sqlite_path"])
        self.max_results = max_results or CONFIG["max_results"]

    def search(self, query, task=None, model="gpt-4o"):
        results = search_iirc(self.sqlite_path, query, self.max_results)
        snippets = [
            f"{item.get('title', '')}: {item.get('body', '')}".strip(": ")
            for item in results
        ]
        rewrite = agent(
            rewrite_search_agent_prompt % (task or query, snippets),
            model=model,
        )
        return response_text(rewrite), snippets


# Compatibility for code that imported the old class name.
DDGSSearch = IIRCSearch
