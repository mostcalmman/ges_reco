"""Sequential batch trainer for long-running model experiments.

This script runs multiple `train.py --model_type ...` jobs one by one,
stores per-model logs, and writes a persistent state file so reruns can
skip completed models.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


MODEL_LIST = [
    "ShuffleNetV2x10",
    "ResNet50_TSM",
    "MobileNetV2_TSM",
    "ShuffleNetV2x10_TSM",
]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run train.py for a fixed model list (long-running friendly)."
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run train.py (default: current interpreter).",
    )
    parser.add_argument(
        "--run_root",
        type=str,
        default="checkpoint/batch_train_runs",
        help=(
            "Root folder for batch artifacts (state file + logs). "
            "Each model checkpoint will be under run_root/<model_type>."
        ),
    )
    parser.add_argument(
        "--rerun_completed",
        action="store_true",
        help="Rerun models marked as completed in state file.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop the entire batch when one model fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running training.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to train.py. Use `--` before them.",
    )
    return parser.parse_args()


def load_or_init_state(state_path: Path) -> dict:
    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {
            "created_at": now_str(),
            "updated_at": now_str(),
            "models": {},
        }

    models = state.setdefault("models", {})
    for model_name in MODEL_LIST:
        models.setdefault(
            model_name,
            {
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "duration_sec": None,
                "exit_code": None,
                "log_file": None,
                "last_cmd": None,
            },
        )

    state["updated_at"] = now_str()
    return state


def save_state(state_path: Path, state: dict) -> None:
    state["updated_at"] = now_str()
    tmp_path = state_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp_path.replace(state_path)


def run_single_model(cmd: list[str], log_path: Path, dry_run: bool) -> int:
    print("=" * 80)
    print(f"[{now_str()}] Running: {' '.join(cmd)}")
    print(f"Log file: {log_path}")

    if dry_run:
        print("Dry-run mode: command not executed.")
        return 0

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"[{now_str()}] CMD: {' '.join(cmd)}\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            print("\nInterrupted. Terminating current training process...")
            log_file.write(f"\n[{now_str()}] Interrupted by user.\n")
            log_file.flush()
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise

        return proc.wait()


def main() -> int:
    args = parse_args()

    run_root = Path(args.run_root)
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    extra_args = list(args.train_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    total = len(MODEL_LIST)

    # Dry-run prints the exact commands without changing persistent state.
    if args.dry_run:
        for idx, model_name in enumerate(MODEL_LIST, start=1):
            model_checkpoint_dir = run_root / model_name
            log_path = logs_dir / f"{idx:02d}_{model_name}.log"

            cmd = [
                args.python,
                "-u",
                "train.py",
                "--model_type",
                model_name,
                "--checkpoint_dir",
                str(model_checkpoint_dir),
            ]
            cmd.extend(extra_args)

            run_single_model(cmd, log_path, dry_run=True)
            print(f"[{idx}/{total}] Dry-run checked: {model_name}")

        print("\nDry-run finished. No state file was changed.")
        return 0

    state_path = run_root / "batch_state.json"
    state = load_or_init_state(state_path)
    save_state(state_path, state)

    for idx, model_name in enumerate(MODEL_LIST, start=1):
        model_state = state["models"][model_name]
        if model_state.get("status") == "completed" and not args.rerun_completed:
            print(f"[{idx}/{total}] Skip completed model: {model_name}")
            continue

        model_checkpoint_dir = run_root / model_name
        log_path = logs_dir / f"{idx:02d}_{model_name}.log"

        cmd = [
            args.python,
            "-u",
            "train.py",
            "--model_type",
            model_name,
            "--checkpoint_dir",
            str(model_checkpoint_dir),
        ]
        cmd.extend(extra_args)

        model_state["status"] = "running"
        model_state["start_time"] = now_str()
        model_state["end_time"] = None
        model_state["duration_sec"] = None
        model_state["exit_code"] = None
        model_state["log_file"] = str(log_path)
        model_state["last_cmd"] = cmd
        save_state(state_path, state)

        begin = time.time()
        try:
            return_code = run_single_model(cmd, log_path, args.dry_run)
        except KeyboardInterrupt:
            model_state["status"] = "interrupted"
            model_state["end_time"] = now_str()
            model_state["duration_sec"] = round(time.time() - begin, 2)
            model_state["exit_code"] = -2
            save_state(state_path, state)
            print("Batch interrupted. Re-run script to continue from unfinished models.")
            return 130

        model_state["end_time"] = now_str()
        model_state["duration_sec"] = round(time.time() - begin, 2)
        model_state["exit_code"] = return_code

        if return_code == 0:
            model_state["status"] = "completed"
            print(f"[{idx}/{total}] Completed: {model_name}")
        else:
            model_state["status"] = "failed"
            print(f"[{idx}/{total}] Failed ({return_code}): {model_name}")
            save_state(state_path, state)
            if args.stop_on_error:
                print("Stop-on-error enabled, aborting batch.")
                return return_code

        save_state(state_path, state)

    failed = [m for m in MODEL_LIST if state["models"][m]["status"] == "failed"]
    completed = [m for m in MODEL_LIST if state["models"][m]["status"] == "completed"]

    print("\nBatch finished.")
    print(f"Completed: {len(completed)}/{total}")
    if failed:
        print(f"Failed models: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
