"""Run attribution methods over a folder of logs and save one JSON line per log.

    python run.py --data "Who&When/Algorithm-Generated" --method all_at_once --sample 50 --limit 10
"""
import argparse
import json
import logging
import os
import random
import time

from tqdm import tqdm

import llm
from data import list_logs, load_log, normalize_agent
from methods import METHODS

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


def main():
    """Runs the chosen methods on each log, one by one, and saves each answer to a results file. Skips logs that already have an answer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Folder of log JSON files")
    parser.add_argument("--method", required=True, help=f"Comma-separated: {', '.join(METHODS)}")
    parser.add_argument("--sample", type=int, default=None, help="Use a fixed random sample of N logs (same files for every dataset folder)")
    parser.add_argument("--files", help="Comma-separated file names to run, e.g. 1.json,4.json (a fixed subset)")
    parser.add_argument("--start", type=int, default=0, help="Index of the first log to run (for batching)")
    parser.add_argument("--limit", type=int, default=None, help="Number of logs to run")
    parser.add_argument("--tag", default="", help="Label for a repeated run; bypasses cached answers")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    llm.cache_tag = args.tag
    methods = [m.strip() for m in args.method.split(",")]
    for m in methods:
        if m not in METHODS:
            parser.error(f"Unknown method '{m}'")

    files = list_logs(args.data)
    if args.files:
        wanted = [f.strip() for f in args.files.split(",")]
        missing = [f for f in wanted if f not in files]
        if missing:
            parser.error(f"Not in {args.data}: {', '.join(missing)}")
        files = [f for f in files if f in wanted]
    if args.sample:
        files = sorted(random.Random(0).sample(files, min(args.sample, len(files))), key=files.index)
    files = files[args.start:args.start + args.limit if args.limit else None]
    dataset = os.path.basename(os.path.normpath(args.data))

    stopped = False
    try:
        for method in methods:
            # "hf.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF:Q4_K_M" -> "Qwen2.5-7B-Instruct-1M-GGUF_Q4_K_M"
            model_dir = llm.MODEL.split("/")[-1].replace(":", "_")
            out_path = os.path.join(args.out, model_dir, dataset, f"{method}{'_' + args.tag if args.tag else ''}.jsonl")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            done = set()
            if os.path.exists(out_path):
                with open(out_path, encoding="utf-8") as f:
                    # Failed logs are tried again on the next run; the newest line for a file is the one that counts
                    latest = {r["file"]: r for r in map(json.loads, filter(str.strip, f))}
                done = {name for name, r in latest.items() if "error" not in r}

            for name in tqdm([f for f in files if f not in done], desc=method):
                log = load_log(os.path.join(args.data, name))
                record = {
                    "file": name,
                    "gt_agent": log["mistake_agent"],
                    "gt_step": int(log["mistake_step"]) if log["mistake_step"] is not None else None,
                }
                started = time.time()
                try:
                    prediction = METHODS[method](log)
                    agent = normalize_agent(prediction["agent"])
                    # Renamed logs: map "Agent_B" back to the original name so runs are comparable (case-insensitive: the model may write "agent_b")
                    agent_map = {k.lower(): v for k, v in log["agent_map"].items()}
                    record.update({
                        "agent": agent_map.get(agent.lower(), agent),
                        "step": prediction["step"],
                        "reason": prediction.get("reason", ""),
                        **({"trace": prediction["trace"]} if "trace" in prediction else {}),
                    })
                except llm.OllamaUnavailable:
                    raise
                except Exception as e:
                    # Kept as a wrong answer so every method is scored on the same logs
                    tqdm.write(f"WARNING {method} failed on {name}: {type(e).__name__}: {e}")
                    record.update({"agent": "Unknown", "step": None, "error": f"{type(e).__name__}: {e}"})
                record["seconds"] = round(time.time() - started, 1)  # for estimating GPU time of the remaining runs
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except llm.OllamaUnavailable as e:
        print(f"\nStopped: {e}")
        stopped = True
    except KeyboardInterrupt:
        print("\nStopped. Re-run the same command to continue where it left off.")
        stopped = True

    print(f"Model requests made this run: {llm.requests_made}")
    # Non-zero exit so a script running many datasets stops instead of failing through the rest
    raise SystemExit(1 if stopped else 0)


if __name__ == "__main__":
    main()
