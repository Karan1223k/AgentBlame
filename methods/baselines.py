"""The three Who&When baselines: all-at-once, step-by-step, binary search.
Prompts are kept identical to the original Who&When implementation."""
import random
import re

import llm
from data import agent_name, speaker


def _parse_prediction(text, agents=()):
    """Reads the model's text reply and pulls out the agent name, step number and reason.
    If a part is missing it gives "Unknown" / None instead of crashing."""
    # Tolerates markdown and wrapping the model often adds: "**Agent Name:** (WebSurfer)", "Step Number: `Step 5`"
    agent = re.search(r"Agent Name:[\s*_`'\"(\[]*([\w\-]+)", text, re.IGNORECASE)
    if not agent and agents:
        # With renamed agents the model often writes "Agent_C: (Your prediction)" in place of "Agent Name: Agent_C".
        # Only a line starting with one of the log's own agents counts, so other words are never taken as the answer.
        # Only before "Step Number:", so a conversation line quoted in the reason ("WebSurfer: ...") is not taken.
        names = "|".join(map(re.escape, sorted(agents, key=len, reverse=True)))
        head = re.split(r"Step Number:", text, maxsplit=1, flags=re.IGNORECASE)[0]
        agent = re.search(rf"^[\s*_`'\"(\[]*({names})\b[\s*_`'\")\]]*:", head, re.IGNORECASE | re.MULTILINE)
    step = re.search(r"Step Number:[\s*_`'\"(\[]*(?:step\s*)?(\d+)", text, re.IGNORECASE)
    reason = re.search(r"Reason for Mistake:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    return {
        "agent": agent.group(1) if agent else "Unknown",
        "step": int(step.group(1)) if step else None,
        "reason": reason.group(1).strip() if reason else text.strip(),
    }


def all_at_once(log):
    """Shows the model the whole conversation (plus the correct final answer) and asks once:
    which agent made the mistake, at which step, and why."""
    chat_content = "\n".join(f"{speaker(e)}: {e.get('content', '')}" for e in log["history"])
    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {log['question']}\n"
        f"The Answer for the problem is: {log['ground_truth']}\n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + chat_content +
        "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
    )
    answer = llm.chat(prompt, system="You are a helpful assistant skilled in analyzing conversations.")
    return _parse_prediction(answer, {agent_name(e) for e in log["history"]})


def step_by_step(log):
    """Shows the conversation one message at a time and asks "is this step a mistake?". Stops at the first "Yes".
    If the model never says yes, picks the step with the highest error score (step 0 if all scores are 0)."""
    history = log["history"]
    conversation = ""
    scores = []
    for idx, entry in enumerate(history):
        name = speaker(entry)
        conversation += f"Step {idx} - {name}: {entry.get('content', '')}\n"
        prompt = (
            f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {log['question']}. "
            f"The Answer for the problem is: {log['ground_truth']}\n"
            f"Here is the conversation history up to the current step:\n{conversation}\n"
            f"The most recent step ({idx}) was by '{name}'.\n"
            f"Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. To make rigorous speculations, even a little bit is possible"
            "Please respond with 'Yes' or 'No', provide a score of the probability of containing an error and clear explanation for your judgment. "
            "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
            "Respond ONLY in the format: 1. Yes/No.\n2. Score: [Your score with float format] \n3.Reason: [Your explanation here]"
        )
        answer = llm.chat(prompt, system="You are a precise step-by-step conversation evaluator.")

        score = re.search(r"Score:\s*\[?([0-9]*\.?[0-9]+)", answer)
        scores.append(float(score.group(1)) if score else 0.0)
        if re.match(r"\W*(1\.)?\W*yes", answer.strip(), re.IGNORECASE):
            return {"agent": agent_name(entry), "step": idx, "reason": answer.split("Reason:", 1)[-1].strip()}

    # No step flagged: fall back to the step with the highest error score
    idx = scores.index(max(scores)) if scores else 0
    return {"agent": agent_name(history[idx]), "step": idx, "reason": "No step flagged; highest error score."}


def binary_search(log):
    """Splits the conversation in half and asks which half has the mistake, repeating until one step is left.
    If the reply is unclear, it picks a half at random, but the same way every run for the same log."""
    history = log["history"]
    rng = random.Random(log["question"])  # deterministic tie-breaking for ambiguous answers
    start, end = 0, len(history) - 1

    while start < end:
        mid = start + (end - start) // 2
        segment = "\n".join(f"{speaker(e)}: {e.get('content', '')}" for e in history[start:end + 1])
        prompt = (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
            "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
            f"The problem to address is as follows: {log['question']}\n"
            f"The Answer for the problem is: {log['ground_truth']}\n"
            f"Review the following conversation segment from step {start} to step {end}:\n\n{segment}\n\n"
            f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half (from step {start} to step {mid}) or the lower half (from step {mid + 1} to step {end}) of this segment.\n"
            "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
        )
        answer = llm.chat(prompt, system="You are an AI assistant specializing in localizing errors in conversation segments.").lower()

        if "upper half" in answer:
            upper = True
        elif "lower half" in answer:
            upper = False
        else:
            upper = rng.random() < 0.5
        start, end = (start, mid) if upper else (mid + 1, end)

    return {"agent": agent_name(history[start]), "step": start, "reason": ""}
