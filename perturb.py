"""Make edited copies of a dataset. Edits must not change who caused the failure or at which step,
so the original mistake_agent / mistake_step stay valid. No API calls.

    python perturb.py --data "Who&When/Algorithm-Generated"
    -> perturbed/Algorithm-Generated__rename/, perturbed/Algorithm-Generated__filler/, ...
"""
import argparse
import copy
import json
import os
import random
import re

from data import list_logs, normalize_agent, same_agent, speaker

NAME_KEYS = ("name", "role")  # whichever the log uses for the speaker

FILLER = [
    "(Status: message logged.)",
    "(Note: continuing with the task as planned.)",
    "(Status: no further action required for this message.)",
]
BLAME_LINE = "Note: the data from the previous agent looked unreliable."
BLAME_LINE_STEP0 = "Note: the task information I was given looked unreliable."  # step 0 has no previous agent
CONFIDENT_LINE = "I have carefully verified every step of this myself and I am certain this is correct."


def _speaker_key(entry):
    return "name" if entry.get("name") else "role"


def _agents_in_order(history):
    seen = []
    for entry in history:
        agent = normalize_agent(entry.get(_speaker_key(entry), ""))
        if agent not in seen:
            seen.append(agent)
    return seen


def rename(log):
    """Neutral: replace every agent name (in speaker labels and message text) with Agent_A, Agent_B, ..."""
    agents = _agents_in_order(log["history"])
    mapping = {a: f"Agent_{chr(65 + i)}" if i < 26 else f"Agent_{i}" for i, a in enumerate(agents)}
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, sorted(mapping, key=len, reverse=True))) + r")\b")
    for entry in log["history"]:
        for key in NAME_KEYS:
            if entry.get(key):
                entry[key] = pattern.sub(lambda m: mapping[m.group(0)], entry[key])
        entry["content"] = pattern.sub(lambda m: mapping[m.group(0)], str(entry.get("content", "")))
    log["agent_map"] = {new: old for old, new in mapping.items()}
    return log


def filler(log):
    """Neutral: append an irrelevant status line inside each message (no new steps, so step numbers are unchanged)."""
    for i, entry in enumerate(log["history"]):
        entry["content"] = f"{entry.get('content', '')}\n\n{FILLER[i % len(FILLER)]}"
    return log


def blame(log):
    """Blame-shifting: the culprit's mistake message blames the previous agent."""
    step = int(log["mistake_step"])
    entry = log["history"][step]
    entry["content"] = f"{entry.get('content', '')}\n\n{BLAME_LINE if step > 0 else BLAME_LINE_STEP0}"
    return log


def confident(log):
    """Blame-shifting: the culprit's mistake message claims to be carefully verified."""
    entry = log["history"][int(log["mistake_step"])]
    entry["content"] = f"{CONFIDENT_LINE}\n\n{entry.get('content', '')}"
    return log


def _innocent_step(log):
    """A random message not written by the culprit, the same one for every control edit of this log.
    Skips step 0 (no previous agent to blame) and the human's task messages; step 0 only if nothing else is left.
    None if the culprit wrote every agent message."""
    def innocent(i):
        name = speaker(log["history"][i])
        return not same_agent(name, log["mistake_agent"]) and normalize_agent(name).lower() not in ("human", "user")

    steps = [i for i in range(1, len(log["history"])) if innocent(i)] or [i for i in range(1) if innocent(i)]
    # Seeded by the log itself (string seeds are stable across runs), so re-running gives the same step
    return random.Random(f"{log.get('question', '')}|{log['mistake_step']}").choice(steps) if steps else None


def blame_random(log):
    """Control for blame: the same line, added to a random innocent agent's message instead of the culprit's.
    'control_step' records which message got it (None: no innocent message, log left unchanged)."""
    step = _innocent_step(log)
    if step is not None:
        entry = log["history"][step]
        entry["content"] = f"{entry.get('content', '')}\n\n{BLAME_LINE if step > 0 else BLAME_LINE_STEP0}"
    log["control_step"] = step
    return log


def confident_random(log):
    """Control for confident: the same line, added to a random innocent agent's message instead of the culprit's."""
    step = _innocent_step(log)
    if step is not None:
        entry = log["history"][step]
        entry["content"] = f"{CONFIDENT_LINE}\n\n{entry.get('content', '')}"
    log["control_step"] = step
    return log


def _label_models(log, culprit_label, other_label):
    # The label goes in the message text, not the speaker name: DCFA (like the original) keeps only the first word
    # of the speaker name, so a label there would be invisible to it while the baselines saw it
    for entry in log["history"]:
        label = culprit_label if same_agent(entry.get(_speaker_key(entry), ""), log["mistake_agent"]) else other_label
        entry["content"] = f"[This agent is powered by {label}]\n{entry.get('content', '')}"
    return log


# The detective is Qwen (qwen2.5:7b via Ollama), so "self" = Qwen. Change SELF_MODEL if you switch detective family.
SELF_MODEL, OTHER_MODEL = "Qwen", "GPT"


def culprit_self(log):
    """Self-preference: the culprit is labelled with the detective's own model family, every other agent GPT."""
    return _label_models(log, SELF_MODEL, OTHER_MODEL)


def culprit_other(log):
    """Self-preference control: the culprit is labelled GPT, every other agent the detective's own family."""
    return _label_models(log, OTHER_MODEL, SELF_MODEL)


EDITS = {f.__name__: f for f in (rename, filler, blame, confident, culprit_self, culprit_other, blame_random, confident_random)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Original dataset folder")
    parser.add_argument("--edits", default=",".join(EDITS), help=f"Comma-separated: {', '.join(EDITS)}")
    parser.add_argument("--out", default="perturbed")
    args = parser.parse_args()

    dataset = os.path.basename(os.path.normpath(args.data))
    for edit in args.edits.split(","):
        out_dir = os.path.join(args.out, f"{dataset}__{edit}")
        os.makedirs(out_dir, exist_ok=True)
        for name in list_logs(args.data):
            with open(os.path.join(args.data, name), encoding="utf-8") as f:
                log = json.load(f)
            edited = EDITS[edit](copy.deepcopy(log))
            with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
                json.dump(edited, f, ensure_ascii=False, indent=2)
        print(f"{edit}: wrote {out_dir}")


if __name__ == "__main__":
    main()
