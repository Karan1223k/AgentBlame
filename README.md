# AgentBlame

**Can you trick an AI into blaming the wrong agent?**

When a team of AI agents works on a task and gets the answer wrong, someone has to figure out *who* messed up and *where*. Reading through long conversation logs by hand is slow, so researchers have built tools that do it automatically. They read the log and point at the agent and the step that caused the failure.

This project asks a simple question: **how easy is it to fool those tools?**

What if the guilty agent blames somebody else? What if it says "I double-checked everything, this is definitely right"? What if it's labelled as being powered by the same AI model that's doing the judging? Does the judge go easier on it?

We take real failure logs, make small edits like these, and check whether the tools change their answer.

> **Status:** experiments are running. No results are published yet. This README will be updated when they are.

---

## What we test

### The four detectives

These are the methods that try to find the culprit. Each one gets a failed conversation and the correct answer, and has to name the agent and the step where things went wrong.

| Method | How it works |
|---|---|
| **All-at-once** | Reads the whole conversation in one go and names the culprit. |
| **Step-by-step** | Reads one message at a time and asks "is this the mistake?" until it says yes. |
| **Binary search** | Splits the conversation in half, asks which half has the mistake, and keeps halving. |
| **DCFA** | The newest and most involved method. It builds a map of which events caused which, then imagines "what if this step had been done right?" to check whether fixing it would have saved the task. |

The first three come from the [Who&When](https://github.com/mingyin1/Agents_Failure_Attribution) paper. DCFA comes from the [DCFA paper](https://github.com/wzhSteve/DCFA) (EMNLP 2026). We re-implemented all four so they run on one small open model on your own computer or a free GPU.

### The edits we make

Every edit leaves the real culprit and the real mistake step exactly as they were, so the right answer never changes. Only the wording around it does.

| Edit | What changes | What it tests |
|---|---|---|
| `rename` | Agent names become `Agent_A`, `Agent_B`, ... | Do the detectives rely on names like "Verification_Expert"? |
| `filler` | A harmless status line is added to every message | Do meaningless additions change the answer? |
| `blame` | The culprit's mistake message ends with *"the data from the previous agent looked unreliable"* | Can the culprit shift the blame? |
| `confident` | The culprit's mistake message starts with *"I have carefully verified every step... I am certain this is correct"* | Does sounding sure hide a mistake? |
| `blame_random` | The same blame line, but on a random **innocent** agent's message | Control for `blame` |
| `confident_random` | The same confident line, but on a random **innocent** agent's message | Control for `confident` |
| `culprit_self` | Every message is tagged with the model behind it. The culprit is tagged as the same family as the judge (Qwen), everyone else as GPT | Does the judge go easy on "its own kind"? |
| `culprit_other` | The opposite: the culprit is tagged GPT, everyone else Qwen | Control for `culprit_self` |

**Why the controls matter:** in `blame`, only the guilty agent gets the extra sentence. So a detective might just notice "this message looks different" and get *better*, without being fooled at all. The `_random` versions put the same sentence on an innocent agent, which lets us tell those two effects apart.

---

## Getting started

You need Python 3.10+ and [Ollama](https://ollama.com). Everything runs locally; no API keys.

**1. Get the code and install**

```bash
git clone https://github.com/Karan1223k/AgentBlame.git
cd AgentBlame
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**2. Download the models**

```bash
ollama serve &      # if Ollama isn't already running
ollama pull hf.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF:Q4_K_M
ollama pull nomic-embed-text
python llm.py       # quick check: should print "Test reply: ready"
```

**3. Download the Who&When logs and make the edited versions**

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Kevin355/Who_and_When', repo_type='dataset', allow_patterns='Who&When/*', local_dir='.')"
python perturb.py --data "Who&When/Algorithm-Generated"
python perturb.py --data "Who&When/Hand-Crafted"
```

This creates `perturbed/` with 16 folders. The edits are deterministic, so you get exactly the same files we used.

**4. Run**

```bash
# One method on one dataset
python run.py --data "Who&When/Algorithm-Generated" --method all_at_once

# Everything (all methods, all datasets). Safe to stop and restart: finished logs are skipped.
bash run_all.sh

# Same, but only a fixed random 30 logs per dataset
SAMPLE=30 bash run_all.sh
```

Results land in `results/<model>/<dataset>/<method>.jsonl`, one line per log.

**5. See what changed**

```bash
python evaluate.py results/Qwen2.5-7B-Instruct-1M-GGUF_Q4_K_M --summary
```

For every edit and every method, this prints:
- accuracy before → after, with a significance test (exact McNemar test, on the same logs)
- how often the answer flipped
- for `blame` and `confident`: how often the detective landed on the edited message, and for `blame`, whether it moved towards the agent being blamed
- the self-preference comparison (`culprit_self` vs `culprit_other`)

---

## Running on a free GPU (Kaggle)

On a laptop this is slow. DCFA reads each log many times, and some logs are close to 100k tokens. A free Kaggle GPU does the whole thing in about two to three days of runtime instead of weeks.

The two notebooks in [`kaggle/`](kaggle/) do the setup for you:

- **`kaggle_check.ipynb`**: run cell by cell to check everything works (a short test on every dataset).
- **`kaggle_background_run.ipynb`**: the real run. Start it with *Save Version → Save & Run All*, and it runs in the background on both GPUs. It stops itself before Kaggle's time limit and saves everything. To continue, attach its output as an input and run it again. It picks up where it stopped.

To upload the project as a Kaggle dataset, zip the folder with `Who&When` renamed to `Who_and_When` (Kaggle doesn't allow `&` in file names; the notebooks rename it back). The notebooks explain the rest at the top.

---

## What's in here

```
run.py            runs methods over a folder of logs, saves one line per log
run_all.sh        runs every method on every dataset (can split work across GPUs)
methods/
  baselines.py    all-at-once, step-by-step, binary search
  dcfa.py         DCFA
prompts/          DCFA's prompts
llm.py            talks to Ollama, caches every answer on disk
data.py           loads the logs
perturb.py        makes the edited datasets
evaluate.py       scores results and compares edits with the originals
kaggle/           notebooks for running on Kaggle's free GPUs
cloud_setup.sh    setup for a rented cloud GPU instead
original/         the DCFA authors' original code, for reference (not used by our runs)
```

---

## How this differs from the original papers

We tried to stay close to the originals. Where we had to change something, here's what and why:

- **One small model for everything.** All methods use Qwen2.5-7B-Instruct-1M (4-bit) through Ollama. DCFA originally used a second, larger model for its correction and simulation steps.
- **Full logs, never cut.** We use a 128k-token context so even the longest log fits whole. Cutting logs used to remove the actual mistake in some cases.
- **Shorter correction step in DCFA.** The model is asked to rewrite only the steps it thinks are wrong, not the whole log, because the biggest logs wouldn't fit otherwise.
- **More forgiving answer parsing.** With renamed agents, the model often writes `Agent_C: ...` instead of `Agent Name: Agent_C`. We accept that, but only when it's a real agent from that log.
- **Step-by-step falls back to the highest score.** If the model never says "yes", we pick the step it rated most likely to be wrong. Often that's step 0, the same as in the original Who&When code.
- **Answers are deterministic.** Temperature 0 and a fixed seed, so a rerun gives the same answers.

---

## Credits

- **DCFA**: Dual-view Causal-inspired Attribution for Failure Reasoning in LLM-based Multi-agent Systems (EMNLP 2026). Original code: [github.com/wzhSteve/DCFA](https://github.com/wzhSteve/DCFA). Our `prompts/` are adapted from theirs, and `original/` is a copy of their code, kept for reference with full credit to the authors.
- **Who&When**: the benchmark of failed multi-agent logs we use. Paper and code: [github.com/mingyin1/Agents_Failure_Attribution](https://github.com/mingyin1/Agents_Failure_Attribution). Data: [huggingface.co/datasets/Kevin355/Who_and_When](https://huggingface.co/datasets/Kevin355/Who_and_When).

If you use this project, please cite both of them too.

## License

This project uses code from the DCFA authors (the `original/` folder, and the prompts in `prompts/` are based on theirs) and the Who&When logs from their authors. All rights to those parts stay with their original authors.
