import argparse
import json
import re
import sqlite3
from pathlib import Path


# Edit these defaults directly before running the script.
CONFIG = {
    "split_input": "benchmarks/iirc/dev.json",
    "flat_output": "benchmarks/iirc/iirc_dev_flat.json",
    "articles_input": "benchmarks/iirc/context_articles.json",
    "sqlite_output": "benchmarks/iirc/context_articles.sqlite3",
    "build_sqlite": True,
    "force": False,
}


def normalize_title(value):
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def normalize_answer(answer):
    answer = answer or {}
    answer_type = answer.get("type")
    if answer_type == "span":
        texts = [
            span.get("text", "").strip()
            for span in answer.get("answer_spans") or []
            if span.get("text", "").strip()
        ]
        return "; ".join(texts)
    if answer_type == "value":
        value = str(answer.get("answer_value", "")).strip()
        unit = str(answer.get("answer_unit", "")).strip()
        return " ".join(part for part in (value, unit) if part)
    if answer_type == "binary":
        return str(answer.get("answer_value", "")).strip().lower()
    if answer_type == "none":
        return "not enough information"
    raise ValueError(f"Unsupported IIRC answer type: {answer_type!r}")


def build_agent_context(question, article):
    links = []
    seen = set()
    for link in article.get("links") or []:
        target = str(link.get("target", "")).strip()
        key = normalize_title(target)
        if target and key not in seen:
            links.append(target)
            seen.add(key)

    link_text = "\n".join(f"- {title}" for title in links) or "- None"
    context = (
        f"Question: {question}\n\n"
        f"Initial article: {article.get('title', '')}\n"
        f"Initial passage:\n{article.get('text', '').strip()}\n\n"
        f"Available linked articles:\n{link_text}"
    )
    return context, links


def flatten_split(input_path, output_path):
    with Path(input_path).open("r", encoding="utf-8") as file:
        articles = json.load(file)

    rows = []
    for article in articles:
        for question in article.get("questions") or []:
            query = question.get("question")
            if not query:
                continue
            agent_context, available_links = build_agent_context(query, article)
            rows.append(
                {
                    "index": question.get("qid"),
                    "source": "allenai/IIRC",
                    "query": query,
                    "answer": normalize_answer(question.get("answer")),
                    "answer_type": (question.get("answer") or {}).get("type"),
                    "original_answer": question.get("answer"),
                    "article_pid": article.get("pid"),
                    "article_title": article.get("title"),
                    "initial_passage": article.get("text"),
                    "available_links": available_links,
                    "agent_context": agent_context,
                    "gold_question_links": question.get("question_links") or [],
                    "gold_context": question.get("context") or [],
                }
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    temporary.replace(output)
    return rows


def iter_json_object(path, chunk_size=1024 * 1024):
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as file:
        buffer = ""
        position = 0
        eof = False

        def read_more():
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = file.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def ensure_data():
            while position >= len(buffer) and not eof:
                read_more()

        def skip_whitespace():
            nonlocal position
            while True:
                ensure_data()
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                read_more()

        def decode_value():
            nonlocal position
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    return value
                except json.JSONDecodeError:
                    if eof:
                        raise
                    read_more()

        read_more()
        skip_whitespace()
        if position >= len(buffer) or buffer[position] != "{":
            raise ValueError(f"{path} is not a top-level JSON object")
        position += 1

        while True:
            skip_whitespace()
            if position < len(buffer) and buffer[position] == "}":
                return
            key = decode_value()
            if not isinstance(key, str):
                raise ValueError("Expected a string key in context article JSON")
            skip_whitespace()
            if position >= len(buffer) or buffer[position] != ":":
                raise ValueError(f"Expected ':' after context article key {key!r}")
            position += 1
            skip_whitespace()
            value = decode_value()
            yield key, value
            skip_whitespace()
            if position < len(buffer) and buffer[position] == ",":
                position += 1
                continue
            if position < len(buffer) and buffer[position] == "}":
                return
            raise ValueError(f"Expected ',' or '}}' after context article {key!r}")


def build_sqlite(articles_path, sqlite_path, force=False):
    source = Path(articles_path)
    output = Path(sqlite_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        with sqlite3.connect(output) as connection:
            count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        print(f"SQLite index already exists: {output} ({count} documents)")
        return count
    if output.exists():
        output.unlink()

    connection = sqlite3.connect(output)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE documents (
                id INTEGER PRIMARY KEY,
                title_key TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        batch = []
        count = 0
        for title, text in iter_json_object(source):
            batch.append((normalize_title(title), title, text))
            if len(batch) >= 500:
                connection.executemany(
                    "INSERT INTO documents(title_key, title, text) VALUES (?, ?, ?)",
                    batch,
                )
                connection.commit()
                count += len(batch)
                batch.clear()
                print(f"indexed documents={count}", flush=True)
        if batch:
            connection.executemany(
                "INSERT INTO documents(title_key, title, text) VALUES (?, ?, ?)",
                batch,
            )
            count += len(batch)
            connection.commit()

        connection.execute(
            """
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                title,
                text,
                content='documents',
                content_rowid='id',
                tokenize='unicode61'
            )
            """
        )
        print("building FTS5 index", flush=True)
        connection.execute(
            "INSERT INTO documents_fts(documents_fts) VALUES ('rebuild')"
        )
        connection.execute(
            "CREATE INDEX documents_title_key_idx ON documents(title_key)"
        )
        connection.execute(
            "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("source_path", str(source.resolve())),
                ("source_size", str(source.stat().st_size)),
                ("document_count", str(count)),
            ],
        )
        connection.commit()
        return count
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(
        description="Flatten IIRC questions and build the local document index."
    )
    parser.add_argument("--split-input", default=CONFIG["split_input"])
    parser.add_argument("--flat-output", default=CONFIG["flat_output"])
    parser.add_argument("--articles-input", default=CONFIG["articles_input"])
    parser.add_argument("--sqlite-output", default=CONFIG["sqlite_output"])
    parser.add_argument(
        "--skip-sqlite",
        action="store_true",
        default=not CONFIG["build_sqlite"],
    )
    parser.add_argument("--force", action="store_true", default=CONFIG["force"])
    args = parser.parse_args()

    rows = flatten_split(args.split_input, args.flat_output)
    print(f"Saved flattened IIRC questions: {args.flat_output} ({len(rows)} rows)")
    if not args.skip_sqlite:
        count = build_sqlite(args.articles_input, args.sqlite_output, args.force)
        print(f"Saved IIRC SQLite index: {args.sqlite_output} ({count} documents)")


if __name__ == "__main__":
    main()
