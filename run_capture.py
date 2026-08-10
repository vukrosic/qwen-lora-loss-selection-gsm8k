#!/usr/bin/env python3
"""Run one serialized command with durable logs and an overwrite-safe receipt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_receipt(path: Path, receipt: dict) -> None:
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def pump(source, destination, visible) -> None:
    for line in iter(source.readline, ""):
        destination.write(line)
        destination.flush()
        visible.write(line)
        visible.flush()
    source.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A child command is required after --")
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run dir: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.run_dir / "receipt.json"
    receipt = {
        "status": "STARTING",
        "command": command,
        "cwd": os.getcwd(),
        "started_at": timestamp(),
        "launcher_pid": os.getpid(),
    }
    write_receipt(receipt_path, receipt)
    started = time.monotonic()
    with (args.run_dir / "stdout.log").open("w") as stdout_file, (
        args.run_dir / "stderr.log"
    ).open("w") as stderr_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        receipt.update({"status": "RUNNING", "child_pid": process.pid})
        write_receipt(receipt_path, receipt)
        threads = [
            threading.Thread(
                target=pump,
                args=(process.stdout, stdout_file, sys.stdout),
                daemon=True,
            ),
            threading.Thread(
                target=pump,
                args=(process.stderr, stderr_file, sys.stderr),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        exit_code = process.wait()
        for thread in threads:
            thread.join()
    receipt.update(
        {
            "status": "COMPLETED" if exit_code == 0 else "FAILED",
            "exit_code": exit_code,
            "ended_at": timestamp(),
            "wall_seconds": time.monotonic() - started,
        }
    )
    write_receipt(receipt_path, receipt)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
