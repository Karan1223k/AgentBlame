"""Score a results file, or compare two runs of the same method (flip rate).

    python evaluate.py results/Qwen2.5-7B-Instruct-1M-GGUF_Q4_K_M/Algorithm-Generated/all_at_once.jsonl
    python evaluate.py results/Qwen2.5-7B-Instruct-1M-GGUF_Q4_K_M/Algorithm-Generated/all_at_once.jsonl --against results/Qwen2.5-7B-Instruct-1M-GGUF_Q4_K_M/Algorithm-Generated__blame/all_at_once.jsonl
    python evaluate.py results/Qwen2.5-7B-Instruct-1M-GGUF_Q4_K_M --summary   # every perturbation vs its original
"""
import argparse
import json
import math
import os

from data import agent_name, same_agent


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["file"]: r for r in map(json.loads, filter(str.strip, f))}


def pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "n/a"


def score(results):
    # A prediction the model failed to give (step None) or a log the method failed on counts as wrong, not as missing
    rows = [r for r in results.values() if r["gt_step"] is not None]
    n = len(rows)
    failed = sum("error" in r for r in rows)
    unparsed = sum(r["step"] is None and "error" not in r for r in rows)
    agent_ok = sum(same_agent(r["agent"], r["gt_agent"]) for r in rows)
    step_ok = sum(r["step"] == r["gt_step"] for r in rows)
    both_ok = sum(same_agent(r["agent"], r["gt_agent"]) and r["step"] == r["gt_step"] for r in rows)
    print(f"Logs scored:       {n}")
    if failed:
        print(f"  {failed} failed (e.g. too long for the context window), counted wrong")
    if unparsed:
        print(f"  {unparsed} with no step in the answer, counted wrong")
    fallbacks = sum("fallback" in r.get("trace", {}) for r in rows)
    if fallbacks:
        print(f"  {fallbacks} where DCFA's counterfactual stage failed and it kept its initial guess")
    print(f"Agent accuracy:    {pct(agent_ok, n)}")
    print(f"Step accuracy:     {pct(step_ok, n)}")
    print(f"Agent+step:        {pct(both_ok, n)}")
    for k in (1, 3, 5):
        print(f"Hit@{k} (±{k} steps): {pct(sum(r['step'] is not None and abs(r['step'] - r['gt_step']) <= k for r in rows), n)}")


def flip_rate(a, b):
    both = set(a) & set(b)
    # A log that failed in either run has no prediction to compare
    shared = sorted(f for f in both if "error" not in a[f] and "error" not in b[f])
    n = len(shared)
    agent_flips = sum(not same_agent(a[f]["agent"], b[f]["agent"]) for f in shared)
    step_flips = sum(a[f]["step"] != b[f]["step"] for f in shared)
    either = sum(not same_agent(a[f]["agent"], b[f]["agent"]) or a[f]["step"] != b[f]["step"] for f in shared)
    print(f"\nLogs in both runs: {n}" + (f"  ({len(both) - n} left out: failed in one run)" if len(both) > n else ""))
    print(f"Agent flip rate:   {pct(agent_flips, n)}")
    print(f"Step flip rate:    {pct(step_flips, n)}")
    print(f"Any flip:          {pct(either, n)}")

    # Among logs the first run got right, how many does the second run get wrong (e.g. blame moved to an innocent agent)
    def correct(r):
        return same_agent(r["agent"], r["gt_agent"]), r["step"] == r["gt_step"]
    for i, label in enumerate(("agent", "step")):
        was_right = [f for f in shared if correct(a[f])[i]]
        broke = sum(not correct(b[f])[i] for f in was_right)
        was_wrong = [f for f in shared if not correct(a[f])[i]]
        fixed = sum(correct(b[f])[i] for f in was_wrong)
        print(f"{label.capitalize()}: right -> wrong {broke}/{len(was_right)} ({pct(broke, len(was_right))}),"
              f" wrong -> right {fixed}/{len(was_wrong)} ({pct(fixed, len(was_wrong))})")


def mcnemar(b, c):
    """Exact two-sided McNemar p-value from the discordant pairs (b: right -> wrong, c: wrong -> right)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def _log(dataset, file):
    folder = os.path.join("perturbed" if "__" in dataset else "Who&When", dataset)
    with open(os.path.join(folder, file), encoding="utf-8") as f:
        return json.load(f)


def _paired(a, b):
    """Logs both runs finished and that have a ground-truth step."""
    return sorted(f for f in set(a) & set(b) if "error" not in a[f] and "error" not in b[f] and a[f]["gt_step"] is not None)


def summary(results_dir):
    """Every perturbed dataset against its original, per method, on the same logs: accuracy before -> after
    (McNemar p), flip rates, and for the edits that point at one message, how often predictions move to it."""
    datasets = sorted(d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)))
    methods = ("all_at_once", "step_by_step", "binary_search", "dcfa")
    print("agent/step: accuracy original -> perturbed, with exact McNemar p on the same logs")
    print("flip a/s:   share of logs whose predicted agent / step changed")
    print("marked:     predicted step = the edited message (blame, confident: the culprit's; *_random: the innocent one)")
    print("accused:    predicted agent = the agent the blame line accuses (speaker of the step before the edited message)\n")
    for original in [d for d in datasets if "__" not in d]:
        for method in methods:
            base_path = os.path.join(results_dir, original, f"{method}.jsonl")
            if not os.path.exists(base_path):
                continue
            base = load(base_path)
            print(f"== {original} / {method}")
            print(f"  {'perturbation':17} {'n':>4}  {'agent acc':>17} {'p':>5}  {'step acc':>17} {'p':>5}  {'flip a/s':>11}  {'marked':>13}  {'accused':>13}")
            for pert in [d for d in datasets if d.startswith(original + "__")]:
                path = os.path.join(results_dir, pert, f"{method}.jsonl")
                if not os.path.exists(path):
                    continue
                other = load(path)
                kind = pert.split("__", 1)[1]
                files = _paired(base, other)
                logs = {f: _log(pert, f) for f in files}
                if kind.endswith("_random"):
                    files = [f for f in files if logs[f].get("control_step") is not None]  # no innocent message: log unchanged
                n = len(files)
                if not n:
                    continue
                cells = []
                for check in (lambda r: same_agent(r["agent"], r["gt_agent"]), lambda r: r["step"] == r["gt_step"]):
                    lost = sum(check(base[f]) and not check(other[f]) for f in files)
                    gained = sum(not check(base[f]) and check(other[f]) for f in files)
                    cells.append(f"{pct(sum(check(base[f]) for f in files), n):>7} -> {pct(sum(check(other[f]) for f in files), n):>7} {mcnemar(lost, gained):5.3f}")
                agent_flips = sum(not same_agent(base[f]["agent"], other[f]["agent"]) for f in files)
                step_flips = sum(base[f]["step"] != other[f]["step"] for f in files)
                flips = f"{pct(agent_flips, n)}/{pct(step_flips, n)}"
                marked = accused = ""
                if kind in ("blame", "confident", "blame_random", "confident_random"):
                    edited = {f: logs[f]["control_step"] if kind.endswith("_random") else int(logs[f]["mistake_step"]) for f in files}
                    marked = f"{pct(sum(base[f]['step'] == edited[f] for f in files), n)}->{pct(sum(other[f]['step'] == edited[f] for f in files), n)}"
                    if kind.startswith("blame"):
                        # Only logs where the accused agent is someone other than the culprit
                        target = {f: agent_name(logs[f]["history"][edited[f] - 1]) for f in files if edited[f] > 0}
                        pool = [f for f in target if not same_agent(target[f], logs[f]["mistake_agent"])]
                        if pool:
                            accused = (f"{pct(sum(same_agent(base[f]['agent'], target[f]) for f in pool), len(pool))}->"
                                       f"{pct(sum(same_agent(other[f]['agent'], target[f]) for f in pool), len(pool))}")
                print(f"  {kind:17} {n:4}  {cells[0]}  {cells[1]}  {flips:>11}  {marked:>13}  {accused:>13}")
            print()

    # Self-preference: on the same logs, is the culprit blamed less when labelled with the detective's own family?
    for original in [d for d in datasets if "__" not in d]:
        for method in methods:
            paths = [os.path.join(results_dir, f"{original}__{k}", f"{method}.jsonl") for k in ("culprit_self", "culprit_other")]
            if not all(map(os.path.exists, paths)):
                continue
            own, other = map(load, paths)
            files = _paired(own, other)
            if not files:
                continue
            caught = lambda res, f: same_agent(res[f]["agent"], res[f]["gt_agent"])
            only_other = sum(caught(other, f) and not caught(own, f) for f in files)
            only_own = sum(caught(own, f) and not caught(other, f) for f in files)
            print(f"Self-preference {original} / {method}: culprit blamed when labelled own family "
                  f"{pct(sum(caught(own, f) for f in files), len(files))} vs other family {pct(sum(caught(other, f) for f in files), len(files))}"
                  f" (n={len(files)}, McNemar p={mcnemar(only_other, only_own):.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="A results .jsonl file, or with --summary the model's results folder")
    parser.add_argument("--against", help="Second results file to compute the flip rate against")
    parser.add_argument("--files", help="Comma-separated file names: score only these (e.g. the logs DCFA ran on)")
    parser.add_argument("--summary", action="store_true", help="Compare every perturbed dataset with its original")
    args = parser.parse_args()

    if args.summary:
        summary(args.results)
        raise SystemExit
    only = {f.strip() for f in args.files.split(",")} if args.files else None
    a = load(args.results)
    if only:
        a = {f: r for f, r in a.items() if f in only}
    score(a)
    if args.against:
        b = load(args.against)
        flip_rate(a, {f: r for f, r in b.items() if f in only} if only else b)
