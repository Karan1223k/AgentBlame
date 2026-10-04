"""DCFA: event extraction -> causal graph -> initial root cause -> counterfactual search -> final root cause.
Uses the one local Ollama model for every stage (the original used a second model for correction and simulation)."""
import json
import logging
import os
import re

import numpy as np

import llm
from data import agent_name

logger = logging.getLogger(__name__)

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts")
MAX_ITERATIONS = 5


def _prompt(name, **fields):
    with open(os.path.join(PROMPT_DIR, name), encoding="utf-8") as f:
        return f.read().format(**fields)


def _step_of(event_id):
    """'event_5_1' -> 5. Also accepts the looser ids the model sometimes returns ('event_5', 5, 'step 5')."""
    match = re.search(r"\d+", str(event_id))
    if not match:
        raise ValueError(f"Not an event id: {event_id!r}")
    return int(match.group(0))


def _event_id(event_id):
    # Every step is a single event here, so any id naming step k means event_k_1
    return f"event_{_step_of(event_id)}_1"


def _graph_json(graph):
    # Compact (no indentation): indented, the graph of a 100+ step log takes ~12k tokens, which the longest logs cannot spare
    return json.dumps(graph, ensure_ascii=False, separators=(",", ":"))


def _dict(value):
    return value if isinstance(value, dict) else {}


def _graph_nodes(graph, event_id):
    """Every entry for event_id in the causal graph. The prompt keys the graph by step ({"8": {"event_dict": {"event_8_1": ...}}}),
    but the model sometimes files events under another key or leaves out the step level, so all of these are searched."""
    nodes = []
    for key, value in graph.items():
        if key == event_id:
            nodes.append(_dict(value))
        nodes.append(_dict(_dict(_dict(value).get("event_dict")).get(event_id)))
    return nodes


def _neighbours(graph, event_id, key, n_steps):
    """The 'cause' or 'effect' events of event_id in the causal graph, as clean ids of real steps."""
    found = []
    listed = [node.get(key) or [] for node in _graph_nodes(graph, event_id)]
    # The model may give one id as a plain string, or (rarely) a number or object instead of a list
    for other in [o for items in listed for o in (items if isinstance(items, list) else [items])]:
        if not isinstance(other, (str, int)):
            continue
        try:
            other = _event_id(other)
        except ValueError:
            continue
        if other != event_id and other not in found and _step_of(other) < n_steps:
            found.append(other)
    return found


def _cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _event_content(value):
    return value.get("content", "") if isinstance(value, dict) else str(value)


# ---------- Stages ----------

def extract_events(log):
    events = {
        str(i): {"agent_name": agent_name(entry), "event_dict": {f"event_{i}_1": entry.get("content", "")}}
        for i, entry in enumerate(log["history"])
    }
    mistakes = llm.chat(_prompt("event_extraction.txt", event_log=json.dumps(events, indent=2, ensure_ascii=False)), as_json=True)
    # The prompt names the task 'mistake_even_mark', and the model sometimes wraps its answer in that key
    if len(mistakes) == 1 and isinstance(next(iter(mistakes.values())), dict) and not next(iter(mistakes)).startswith("event_"):
        mistakes = next(iter(mistakes.values()))
    return events, {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in mistakes.items()}


def build_causal_graph(events, mistakes):
    return llm.chat(_prompt("event_causality.txt", ALL_EVENT_JSON=events, MISTAKE_EVENTS_REASONS_JSON=mistakes), as_json=True)


def initial_root_cause(log, events, mistakes, graph):
    return llm.chat(_prompt(
        "failure_attribution.txt",
        QUESTION=log["question"], GROUND_TRUTH=log["ground_truth"],
        ALL_EVENTS_JSON=json.dumps(events, indent=2, ensure_ascii=False), MISTAKE_EVENTS_REASONS_JSON=json.dumps(mistakes, indent=2, ensure_ascii=False),
        CAUSAL_GRAPH_JSON=_graph_json(graph),
    ), as_json=True)


def corrected_log(log, events, mistakes, graph):
    template = "root_cause_event_correction_hand-crafted.txt" if log["handcrafted"] else "root_cause_event_correction.txt"
    reply = llm.chat(_prompt(
        template,
        QUESTION=log["question"], GROUND_TRUTH=log["ground_truth"],
        MISTAKE_EVENT_REASONS=json.dumps(mistakes, indent=2, ensure_ascii=False), ALL_EVENTS=json.dumps(events, indent=2, ensure_ascii=False),
        CAUSAL_GRAPH_JSON=_graph_json(graph),
    ), as_json=True)
    # The prompt asks for {"event_5_1": "corrected text"}; the model may still use the original nested
    # {"5": {"event_dict": {"event_5_1": ...}}} form. Both become the nested form used by total_effect.
    corrected = {}
    for key, value in reply.items():
        nested = _dict(_dict(value).get("event_dict"))
        for event_id, content in (nested.items() if nested else [(key, value)]):
            try:
                step = _step_of(event_id)
            except ValueError:
                continue
            if str(step) in events:
                corrected.setdefault(str(step), {"event_dict": {}})["event_dict"][_event_id(event_id)] = content
    return corrected


def simulate(log, steps, root_cause):
    return llm.chat(_prompt(
        "simulation_outcome.txt",
        QUESTION=log["question"], MODIFIED_LOG=json.dumps(steps, indent=2, ensure_ascii=False), ROOT_CAUSE_EVENT=json.dumps(root_cause, indent=2),
    ), as_json=True)


def final_root_cause(log, events, mistakes, useful_events):
    useful_events = dict(sorted(useful_events.items(), key=lambda kv: _step_of(kv[0])))
    # In step order like the original. A set here would change order between runs (Python randomises string hashes),
    # and since each update() overwrites the previous step, the prompt would differ from run to run.
    steps = sorted({_step_of(e) for e in useful_events})
    causal_events = {}
    for step in map(str, steps):
        if step in events:
            causal_events.update(events[step])
            event_id = f"event_{step}_1"
            if event_id in mistakes:
                useful_events[event_id] = useful_events.get(event_id, "") + "\n" + mistakes[event_id]
    return llm.chat(_prompt(
        "final_failure_attribution.txt",
        QUESTION=log["question"], GROUND_TRUTH=log["ground_truth"],
        CAUSAL_EVENTS_JSON=json.dumps(causal_events, indent=2, ensure_ascii=False), MISTAKE_EVENTS_REASONS_JSON=json.dumps(useful_events, indent=2, ensure_ascii=False),
    ), as_json=True)


# ---------- Counterfactual search ----------

def total_effect(log, events, graph, corrected, candidate):
    """How much closer to the ground truth the simulated answer gets when `candidate` (and everything before it) is corrected,
    compared with correcting only the steps before it."""
    k = _step_of(candidate)
    if candidate not in _dict(_dict(corrected.get(str(k))).get("event_dict")):
        return 0.0, ""

    def mixed_log(last_corrected_step):
        steps = []
        for step in sorted(events, key=int):
            fixed = _dict(_dict(corrected.get(step)).get("event_dict"))
            source = fixed if int(step) <= last_corrected_step and fixed else events[step]["event_dict"]
            content = " ".join(_event_content(v) for v in source.values())
            steps.append({"step_id": step, "agent": events[step]["agent_name"], "content": content})
        return steps

    effects = _neighbours(graph, candidate, "effect", len(events))
    with_fix = simulate(log, mixed_log(k), {candidate: effects})
    without_fix = simulate(log, mixed_log(k - 1), None)

    gt = llm.embed(log["ground_truth"])
    effect = _cosine(llm.embed(str(with_fix.get("answer", ""))), gt) - _cosine(llm.embed(str(without_fix.get("answer", ""))), gt)
    return effect, with_fix.get("reason", "")


def search(log, events, graph, corrected, start_event, direction, visited):
    """Walk the causal graph from start_event (forward = towards causes, backward = towards effects),
    moving to the neighbour whose correction has the largest positive effect."""
    useful, current, prev_step = {}, start_event, _step_of(start_event)
    for _ in range(MAX_ITERATIONS):
        neighbours = _neighbours(graph, current, "cause" if direction == "forward" else "effect", len(events))
        best, best_effect = None, -float("inf")
        for candidate in neighbours:
            if candidate in visited:
                continue
            visited.add(candidate)
            try:
                effect, reason = total_effect(log, events, graph, corrected, candidate)
            except llm.OllamaUnavailable:
                raise
            except Exception as e:
                # The original also skips a candidate it cannot evaluate
                logger.info(f"Skipping candidate {candidate}: {e}")
                continue
            e, b = round(effect, 3), round(best_effect, 3)
            earlier = direction == "forward" and _step_of(candidate) < prev_step
            later = direction == "backward" and _step_of(candidate) > prev_step
            if e > 0 and (e > b or (e == b and (earlier or later))):
                best, best_effect, prev_step = (candidate, reason), effect, _step_of(candidate)
        if not best:
            break
        current = best[0]
        useful[current] = best[1]
    return useful


def dcfa(log):
    events, mistakes = extract_events(log)
    graph = build_causal_graph(events, mistakes)
    initial = initial_root_cause(log, events, mistakes, graph)
    # The schema also asks for "event_step"; use it if the model left out the event id
    start_event = _event_id(initial.get("root_cause_event") or initial["event_step"])
    trace = {"initial": start_event}

    try:
        corrected = corrected_log(log, events, mistakes, graph)
        visited = set()
        useful = {start_event: "It's the initial root cause event selected by the LLM. You need to check it and find the true root cause event."}
        useful.update(search(log, events, graph, corrected, start_event, "forward", visited))
        useful.update(search(log, events, graph, corrected, start_event, "backward", visited))
        trace["candidates"] = list(useful)
        final = final_root_cause(log, events, mistakes, useful)
        root, reason = _event_id(final["root_cause_event"]), final.get("reason", "")
    except llm.OllamaUnavailable:
        raise
    except Exception as e:
        # Same fallback as the original: keep the initial guess if the counterfactual stage fails
        logger.warning(f"Counterfactual stage failed ({e}); using initial root cause")
        trace["fallback"] = f"{type(e).__name__}: {e}"
        root, reason = start_event, initial.get("reason", "")

    step = min(_step_of(root), len(log["history"]) - 1)
    return {"agent": agent_name(log["history"][step]), "step": step, "reason": reason, "trace": trace}
