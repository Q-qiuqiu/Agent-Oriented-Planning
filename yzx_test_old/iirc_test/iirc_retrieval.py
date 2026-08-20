import re
import sqlite3
from pathlib import Path


def normalize_title(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def query_terms(query):
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", query or "")
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
    filtered = [term.casefold() for term in terms if term.casefold() not in stopwords]
    return filtered[:20] or [term.casefold() for term in terms[:20]]


def fts_query(query):
    terms = query_terms(query)
    return " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms)


def result_record(title, text, rank=None, max_chars=3500):
    body = re.sub(r"<[^>]+>", " ", text or "")
    body = re.sub(r"\s+", " ", body).strip()
    return {
        "title": title,
        "href": f"iirc://{normalize_title(title)}",
        "body": body[:max_chars],
        "rank": rank,
        "_backend": "iirc_sqlite",
    }


def search_iirc(sqlite_path, query, max_results=5):
    database = Path(sqlite_path)
    if not database.exists():
        raise FileNotFoundError(
            f"IIRC SQLite index not found: {database}. "
            "Run python3 iirc_test/prepare_iirc.py first."
        )

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        exact = connection.execute(
            "SELECT title, text FROM documents WHERE title_key = ?",
            (normalize_title(query),),
        ).fetchone()
        results = []
        seen = set()
        if exact:
            results.append(result_record(exact["title"], exact["text"], rank=-1.0))
            seen.add(normalize_title(exact["title"]))

        expression = fts_query(query)
        if expression:
            rows = connection.execute(
                """
                SELECT d.title, d.text, bm25(documents_fts, 4.0, 1.0) AS rank
                FROM documents_fts
                JOIN documents AS d ON d.id = documents_fts.rowid
                WHERE documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (expression, max_results * 3),
            ).fetchall()
            for row in rows:
                key = normalize_title(row["title"])
                if key in seen:
                    continue
                results.append(result_record(row["title"], row["text"], row["rank"]))
                seen.add(key)
                if len(results) >= max_results:
                    break
        return results
