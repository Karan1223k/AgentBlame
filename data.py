"""Loading Who&When logs."""
import json
import os
import re


def list_logs(directory):
    """Gets all the log file names in a folder, sorted by number (1, 2, 3... not 1, 10, 11)."""
    files = [f for f in os.listdir(directory) if f.endswith(".json")]
    return sorted(files, key=lambda f: int(re.sub(r"\D", "", f) or 0))


def load_log(path):
    """Opens one log file and returns its contents in the same format for both datasets."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "question": data.get("question", ""),
        # Hand-Crafted logs use "groundtruth", Algorithm-Generated logs use "ground_truth"
        "ground_truth": str(data.get("ground_truth", data.get("groundtruth", ""))),
        # Never shortened: the model's 128k-token context holds the longest log (~87k tokens) whole
        "history": data.get("history", []),
        "mistake_agent": data.get("mistake_agent"),
        "mistake_step": data.get("mistake_step"),
        # Hand-Crafted (Magentic-One) logs only have "role"; Algorithm-Generated logs have "name"
        "handcrafted": "Hand-Crafted" in path,
        # Set by perturb.py when agents are renamed: {new name: original name}
        "agent_map": data.get("agent_map", {}),
    }


def speaker(entry):
    """Tells who said this message, exactly as written in the log, e.g. 'Orchestrator (thought)'."""
    return entry.get("name") or entry.get("role") or "Unknown"


def agent_name(entry):
    """Tells which agent said this message, as a clean name, e.g. 'Orchestrator'."""
    return normalize_agent(speaker(entry))


def normalize_agent(name):
    """Cleans up an agent name by keeping only the first word, e.g. 'Orchestrator (thought)' -> 'Orchestrator'."""
    match = re.match(r"[\w\-]+", str(name).strip())
    return match.group(0) if match else "Unknown"


def same_agent(a, b):
    """Checks if two names mean the same agent, ignoring capital letters ('Websurfer' and 'WebSurfer' match)."""
    return normalize_agent(a).lower() == normalize_agent(b).lower()
