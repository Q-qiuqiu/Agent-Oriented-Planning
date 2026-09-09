#!/usr/bin/env python3
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path


def state_path(state_dir, model):
    safe_name = model.replace("/", "_")
    return Path(state_dir) / f"{safe_name}.json"


def process_stat(pid):
    try:
        content = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    closing_paren = content.rfind(")")
    if closing_paren < 0:
        return None
    fields_after_comm = content[closing_paren + 2 :].split()
    if len(fields_after_comm) <= 19:
        return None
    return {"state": fields_after_comm[0], "start_ticks": fields_after_comm[19]}


def process_start_ticks(pid):
    stat = process_stat(pid)
    return stat["start_ticks"] if stat else None


def process_tty(pid):
    try:
        return os.readlink(f"/proc/{pid}/fd/0")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return "-"


def read_state(path):
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def is_running(state):
    try:
        pid = int(state["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    stat = process_stat(pid)
    return (
        stat is not None
        and stat["state"] != "Z"
        and str(stat["start_ticks"]) == str(state.get("process_start_ticks"))
    )


def write_state(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def register(args):
    ticks = process_start_ticks(args.pid)
    if ticks is None:
        raise SystemExit(f"Cannot register missing process PID {args.pid}")
    record = {
        "model": args.model,
        "model_path": args.model_path,
        "pid": args.pid,
        "process_start_ticks": ticks,
        "cuda_visible_devices": args.gpus,
        "port": args.port,
        "runner": args.runner,
        "mode": args.mode,
        "log_file": args.log_file,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_state(state_path(args.state_dir, args.model), record)


def load_states(state_dir, model=None):
    directory = Path(state_dir)
    paths = [state_path(directory, model)] if model else sorted(directory.glob("*.json"))
    rows = []
    for path in paths:
        state = read_state(path)
        if state is None:
            continue
        state["status"] = "running" if is_running(state) else "stale"
        state["tty"] = process_tty(state.get("pid")) if state["status"] == "running" else "-"
        rows.append(state)
    return rows


def print_table(rows):
    headers = ("MODEL", "PID", "GPU", "PORT", "STATUS", "MODE", "TTY", "LOG")
    values = [
        (
            str(row.get("model", "-")),
            str(row.get("pid", "-")),
            str(row.get("cuda_visible_devices", "-")),
            str(row.get("port", "-")),
            str(row.get("status", "-")),
            str(row.get("mode", "-")),
            str(row.get("tty", "-")),
            str(row.get("log_file") or "-"),
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for row in values:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    template = "  ".join(f"{{:<{width}}}" for width in widths)
    print(template.format(*headers))
    print(template.format(*("-" * width for width in widths)))
    for row in values:
        print(template.format(*row))
    if not rows:
        print("No registered model servers.")


def status(args):
    rows = load_states(args.state_dir, args.model)
    if not args.all:
        rows = [row for row in rows if row["status"] == "running"]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print_table(rows)
    return 0


def check_running(args):
    rows = load_states(args.state_dir, args.model)
    return 0 if any(row["status"] == "running" for row in rows) else 1


def stop(args):
    path = state_path(args.state_dir, args.model)
    state = read_state(path)
    if state is None:
        print(f"No registered server for model {args.model!r}.", file=sys.stderr)
        return 1
    if not is_running(state):
        path.unlink(missing_ok=True)
        print(f"Removed stale state for model {args.model!r}.")
        return 0

    pid = int(state["pid"])
    print(
        f"Stopping {args.model} | pid={pid} | "
        f"gpu={state.get('cuda_visible_devices')} | port={state.get('port')}",
        flush=True,
    )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        path.unlink(missing_ok=True)
        return 0

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if not is_running(state):
            path.unlink(missing_ok=True)
            print("Stopped.")
            return 0
        time.sleep(0.25)

    if not args.force:
        print(
            f"PID {pid} is still running after {args.timeout:.1f}s. "
            "Retry with --force to send SIGKILL.",
            file=sys.stderr,
        )
        return 2

    if is_running(state):
        os.kill(pid, signal.SIGKILL)
    for _ in range(20):
        if not is_running(state):
            path.unlink(missing_ok=True)
            print("Force-stopped.")
            return 0
        time.sleep(0.25)
    print(f"PID {pid} did not exit after SIGKILL.", file=sys.stderr)
    return 2


def build_parser():
    parser = argparse.ArgumentParser(description="Track and stop local model servers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--state-dir", required=True)
    register_parser.add_argument("--model", required=True)
    register_parser.add_argument("--model-path", required=True)
    register_parser.add_argument("--pid", required=True, type=int)
    register_parser.add_argument("--gpus", required=True)
    register_parser.add_argument("--port", required=True, type=int)
    register_parser.add_argument("--runner", required=True)
    register_parser.add_argument("--mode", choices=("foreground", "background"), required=True)
    register_parser.add_argument("--log-file", default="")
    register_parser.set_defaults(handler=register)

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--state-dir", required=True)
    status_parser.add_argument("--model")
    status_parser.add_argument("--json", action="store_true")
    status_parser.add_argument("--all", action="store_true")
    status_parser.set_defaults(handler=status)

    running_parser = subparsers.add_parser("is-running")
    running_parser.add_argument("--state-dir", required=True)
    running_parser.add_argument("--model", required=True)
    running_parser.set_defaults(handler=check_running)

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--state-dir", required=True)
    stop_parser.add_argument("--model", required=True)
    stop_parser.add_argument("--timeout", type=float, default=30.0)
    stop_parser.add_argument("--force", action="store_true")
    stop_parser.set_defaults(handler=stop)
    return parser


def main():
    args = build_parser().parse_args()
    result = args.handler(args)
    raise SystemExit(result or 0)


if __name__ == "__main__":
    main()
