"""Local Ollama client shared by every method, with a disk cache so repeated prompts are free."""
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Qwen2.5-7B trained for long input (up to 1M tokens), at the same Q4_K_M quantization as Ollama's qwen2.5:7b
MODEL = os.getenv("OLLAMA_MODEL", "hf.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF:Q4_K_M")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
# Holds DCFA's biggest prompt on the longest log (~87k-token log + graph + instructions) with room left for the reply.
# Ollama's default is much smaller and silently cuts prompts.
CONTEXT_TOKENS = int(os.getenv("OLLAMA_CONTEXT_TOKENS", "131072"))
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "llm")

RETRY_WAITS = [10, 30, 60, 120, 300]  # seconds to wait before each retry when Ollama is unreachable (~8 min in total)

logger = logging.getLogger(__name__)
cache_tag = ""  # set by run.py --tag; a different tag means fresh (uncached) answers
requests_made = 0


class OllamaUnavailable(Exception):
    pass


class PromptTooLong(Exception):
    """The prompt does not fit in the context window. Ollama would otherwise silently cut it."""


def _post(path, body, read):
    """Sends a request to Ollama and returns read(response). Turns connection problems into clear errors.
    Waits and retries for a few minutes when Ollama is unreachable or crashed (e.g. restarting), so an unattended run survives it."""
    global requests_made
    requests_made += 1
    request = urllib.request.Request(OLLAMA_URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for wait in RETRY_WAITS + [None]:
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                return read(response)
        except urllib.error.HTTPError as e:
            message = e.read().decode()[:300]
            if "exceed" in message and "context" in message:
                raise PromptTooLong(f"Prompt is longer than the {CONTEXT_TOKENS}-token context window")
            if e.code < 500 or wait is None:
                raise OllamaUnavailable(f"Ollama error {e.code}: {message}")
            problem = f"Ollama error {e.code}: {message}"
        except (urllib.error.URLError, ConnectionError) as e:
            # ConnectionError: Ollama died mid-request (e.g. out of GPU memory) and dropped the connection
            reason = getattr(e, "reason", None) or f"{type(e).__name__}: {e}"
            if wait is None:
                raise OllamaUnavailable(f"Cannot reach Ollama at {OLLAMA_URL}. Is Ollama running? ({reason})")
            problem = f"Cannot reach Ollama ({reason})"
        logger.warning(f"{problem}; retrying in {wait}s")
        time.sleep(wait)


def _looping(text):
    """True if the reply has ended in a loop: the same piece (up to 400 characters, e.g. a newline or a repeated sentence)
    repeated over at least the last 2000 characters. A 7B model sometimes does this, and with no reply cap it would
    run until the context is full. No log in Who&When contains such a run, so copying a log never triggers it."""
    tail = text[-4000:]
    for period in range(1, 401):
        span = max(2000, 10 * period)
        if len(tail) >= span and tail[-span:] == (tail[-period:] * (span // period + 1))[-span:]:
            return True
    return False


def _read_stream(response):
    """Reads a streamed chat reply. Stops early (closing the connection, which stops Ollama) if the reply is looping."""
    parts, chunks = [], 0
    for line in response:
        if not line.strip():
            continue
        data = json.loads(line)
        if "error" in data:
            raise RuntimeError(f"Ollama error: {data['error'][:300]}")
        parts.append(data.get("message", {}).get("content", ""))
        chunks += 1
        if data.get("done"):
            if data.get("done_reason") == "length":
                logger.warning("Reply hit the context window and was cut")
            break
        if chunks % 50 == 0 and _looping("".join(parts)):
            logger.warning("Reply was looping; stopped it early")
            break
    return "".join(parts)


def _cached(kind, payload, call):
    """Returns the saved answer if this exact request was made before; otherwise asks Ollama and saves the answer to disk."""
    key = hashlib.sha256(json.dumps([kind, cache_tag, payload], sort_keys=True).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, key[:2], key + ".json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    result = call()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Write then rename, so stopping mid-write never leaves a half-written (unreadable) cache file.
    # The temp name is per process, so parallel runs saving the same answer (e.g. a shared embedding) do not collide.
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    os.replace(tmp, path)
    return result


def chat(prompt, system=None, as_json=False):
    """Asks the model a question and returns its answer, as plain text or as JSON (a dict) when as_json=True."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]

    def call():
        body = {
            "model": MODEL,
            "messages": messages,
            "stream": True,  # lets _read_stream stop a looping reply
            "think": False,
            "truncate": False,  # error instead of silently dropping part of the prompt
            # The reply may use all the context the prompt leaves. shift=False makes Ollama stop there instead of shifting
            # the context (which silently loses the start of the prompt). Some logs are only ~2 characters per token,
            # so a reply limit computed from the prompt's length cannot guarantee this on its own.
            "shift": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": CONTEXT_TOKENS, "num_predict": -1},
        }
        if as_json:
            body["format"] = "json"
        return _post("/api/chat", body, _read_stream)

    text = _cached("chat", [MODEL, CONTEXT_TOKENS, messages, as_json], call)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if not as_json:
        return text
    result = parse_json(text)
    if not isinstance(result, dict):
        raise ValueError(f"Model returned JSON that is not an object: {text[:200]}")
    return result


def embed(text):
    """Turns text into a list of numbers (an embedding) so two pieces of text can be compared for similarity."""
    text = text.replace("\n", " ").strip() or " "
    body = {"model": EMBED_MODEL, "input": [text]}
    return _cached("embed", [EMBED_MODEL, text], lambda: _post("/api/embed", body, lambda r: json.loads(r.read()))["embeddings"][0])


def parse_json(text):
    """Pulls the JSON out of the model's reply, even if the model wrapped it in ``` or extra words."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    if start != -1:
        cut = _complete_part(text[start:])
        if cut is not None:
            return cut
    raise ValueError(f"Model did not return JSON: {text[:200]}")


def _complete_part(text):
    """For a reply cut off mid-JSON (context full, or stopped while looping): keeps every complete entry and closes the
    open brackets, e.g. '{"a": "x", "b": "unfini' -> {"a": "x"}. Returns None if nothing complete is left."""
    closers = {"{": "}", "[": "]"}
    stack, in_string, escaped, cut_points = [], False, False, []
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in closers:
            stack.append(closers[ch])
        elif ch in "}]":
            if not stack:
                break
            stack.pop()
            if not stack:
                break
        elif ch == ",":
            # Everything before this comma is complete; closing the open brackets here gives valid JSON
            cut_points.append((i, "".join(reversed(stack))))
    for i, closing in reversed(cut_points[-200:]):
        try:
            return json.loads(text[:i] + closing)
        except json.JSONDecodeError:
            continue
    return None


if __name__ == "__main__":
    # Check the setup: list installed models and send one tiny test prompt
    installed = [m["name"] for m in json.loads(urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=10).read())["models"]]
    print("Installed models:", ", ".join(installed) or "none")
    for needed in (MODEL, EMBED_MODEL):
        if not any(name == needed or name.split(":")[0] == needed for name in installed):
            print(f"Missing: run `ollama pull {needed}`")
    print("Test reply:", chat("Reply with just the word: ready"))
