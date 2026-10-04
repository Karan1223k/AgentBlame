#!/bin/bash
# One-time setup on the rented GPU machine (Ubuntu with an NVIDIA driver, e.g. GCP L4 or RunPod L40S).
#   bash cloud_setup.sh
set -euo pipefail
cd "$(dirname "$0")"

nvidia-smi --query-gpu=name,memory.total --format=csv

if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
ollama --version  # llm.py needs a recent Ollama (the "truncate" and "shift" request options)

# One request at a time: each slot holds a full 128k-token KV cache (~7 GB). Keep the model loaded between requests.
# On a machine without systemd (e.g. RunPod) start it yourself instead:
#   OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=-1 nohup ollama serve > ollama.log 2>&1 &
if command -v systemctl >/dev/null && systemctl is-active --quiet ollama; then
  sudo mkdir -p /etc/systemd/system/ollama.service.d
  printf '[Service]\nEnvironment="OLLAMA_NUM_PARALLEL=1"\nEnvironment="OLLAMA_KEEP_ALIVE=-1"\n' | sudo tee /etc/systemd/system/ollama.service.d/agentblame.conf
  sudo systemctl daemon-reload && sudo systemctl restart ollama
elif ! curl -s localhost:11434 >/dev/null; then
  OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=-1 nohup ollama serve > ollama.log 2>&1 &
fi
for i in $(seq 30); do curl -s localhost:11434 >/dev/null && break; sleep 1; done

set -a; source .env; set +a
ollama pull "$OLLAMA_MODEL"
ollama pull "$OLLAMA_EMBED_MODEL"

python3 -m venv venv
venv/bin/pip install -q -r requirements.txt

venv/bin/python llm.py   # lists models and sends one test prompt
ollama ps                # PROCESSOR must say 100% GPU; anything on CPU will be far slower
