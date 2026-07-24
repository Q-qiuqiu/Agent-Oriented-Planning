from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ddgs import DDGS

from planner import agent, response_text
from prompt import rewrite_search_agent_prompt


BACKENDS = [
    "duckduckgo",
    "brave",
    "startpage",
    "bing",
    "yahoo",
    "yandex",
]


def search_one_backend(
    query: str,
    backend: str,
    results_per_backend: int,
    proxy: str | None,
    timeout: int,
    retries: int,
    region: str,
) -> dict[str, Any]:
    last_error: str | None = None

    for attempt in range(retries):
        try:
            results = DDGS(proxy=proxy, timeout=timeout).text(
                query=query,
                region=region,
                safesearch="moderate",
                max_results=results_per_backend,
                backend=backend,
            )
            backend_results = list(results or [])
            formatted_results = []

            for rank, item in enumerate(backend_results[:results_per_backend], start=1):
                url = item.get("href")
                if not url:
                    continue
                formatted_results.append(
                    {
                        "title": item.get("title", ""),
                        "href": url,
                        "body": item.get("body", ""),
                        "backend": backend,
                        "rank": rank,
                    }
                )

            return {"backend": backend, "results": formatted_results, "error": None}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries - 1:
                delay = (2**attempt) + random.uniform(0.2, 0.8)
                time.sleep(delay)

    return {"backend": backend, "results": [], "error": last_error}


def search_web(
    query: str,
    results_per_backend: int = 10,
    final_max_results: int = 10,
    max_workers: int = 3,
    proxy: str | None = None,
    timeout: int = 30,
    retries: int = 3,
    region: str = "us-en",
    backends: list[str] | None = None,
) -> dict[str, Any]:
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty")

    if isinstance(backends, str):
        backends = [item.strip() for item in backends.split(",") if item.strip()]
    backends = list(backends or BACKENDS)
    raw_results: list[dict[str, Any]] = []
    backend_status: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_backend = {
            executor.submit(
                search_one_backend,
                query,
                backend,
                results_per_backend,
                proxy,
                timeout,
                retries,
                region,
            ): backend
            for backend in backends
        }

        for future in as_completed(future_to_backend):
            backend = future_to_backend[future]
            try:
                output = future.result()
                backend_results = output["results"]
                error = output["error"]
            except Exception as exc:
                backend_results = []
                error = f"{type(exc).__name__}: {exc}"

            raw_results.extend(backend_results)
            backend_status[backend] = {
                "success": error is None,
                "result_count": len(backend_results),
                "error": error,
            }
            if error is not None:
                errors[backend] = error

    merged_results: dict[str, dict[str, Any]] = {}
    for item in raw_results:
        url = item["href"]
        backend = item["backend"]
        rank = item["rank"]

        if url not in merged_results:
            merged_results[url] = {
                "title": item["title"],
                "href": url,
                "body": item["body"],
                "backends": [],
                "backend_ranks": {},
            }

        merged_item = merged_results[url]
        old_rank = merged_item["backend_ranks"].get(backend)
        if old_rank is None:
            merged_item["backends"].append(backend)
            merged_item["backend_ranks"][backend] = rank
        else:
            merged_item["backend_ranks"][backend] = min(old_rank, rank)

    sorted_results = list(merged_results.values())
    for item in sorted_results:
        ranks = list(item["backend_ranks"].values())
        item["backend_count"] = len(item["backends"])
        item["best_rank"] = min(ranks)
        item["average_rank"] = sum(ranks) / len(ranks)

    sorted_results.sort(
        key=lambda item: (
            -item["backend_count"],
            item["average_rank"],
            item["best_rank"],
            item["href"],
        )
    )

    return {
        "query": query,
        "results": sorted_results[:final_max_results],
        "raw_result_count": len(raw_results),
        "unique_result_count": len(sorted_results),
        "backend_status": backend_status,
        "errors": errors,
    }


class DDGSSearch:
    def __init__(
        self,
        results_per_backend=10,
        final_max_results=10,
        max_workers=3,
        proxy="http://10.134.110.145:7890",
        timeout=30,
        retries=3,
        region="us-en",
        backends=None,
    ):
        self.options = {
            "results_per_backend": results_per_backend,
            "final_max_results": final_max_results,
            "max_workers": max_workers,
            "proxy": proxy,
            "timeout": timeout,
            "retries": retries,
            "region": region,
            "backends": backends,
        }

    def search(self, query):
        output = search_web(query, **self.options)
        if not output["results"]:
            raise RuntimeError(f"DDGS returned no results: {output['errors']}")
        snippets = [
            f"{item.get('title', '')}: {item.get('body', '')}".strip(": ")
            for item in output["results"]
        ]
        rewrite = agent(rewrite_search_agent_prompt % (query, snippets))
        return response_text(rewrite)
