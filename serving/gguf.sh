#!/usr/bin/env bash
# S1: a GGUF file for llama.cpp (llama-server, llama-cli), LM Studio and Ollama, from an export made by
# serving/export_hf.py. Fetches llama.cpp's converter (only the converter and its gguf library, pinned to the commit
# Homebrew's llama.cpp 1.0.0 was built from) into ~/.cache/tinystories-arm/llama.cpp the first time, converts to
# bfloat16, then quantizes to 8 bits with Homebrew's llama-quantize.
#
#   serving/gguf.sh exports/chat-mini-fineweb-ed-202611152150-2746ba-qwen3 fineweb-edu-124m-chat-dpo
#   llama-server -m exports/gguf/fineweb-edu-124m-chat-dpo-Q8_0.gguf --jinja -c 512 --port 8080   # OpenAI API on :8080
set -euo pipefail
export_dir=${1:?usage: serving/gguf.sh <export folder> <name>}
name=${2:?usage: serving/gguf.sh <export folder> <name>}
root=$(cd "$(dirname "$0")/.." && pwd)
commit=b1b770d90
cache=${LLAMA_CPP_CONVERTER:-$HOME/.cache/tinystories-arm/llama.cpp}
python=${PYTHON:-$root/.venv/bin/python}

if [ ! -f "$cache/convert_hf_to_gguf.py" ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp.git "$cache"
  git -C "$cache" sparse-checkout set --no-cone /convert_hf_to_gguf.py /conversion/ /gguf-py/
  git -C "$cache" checkout -q "$commit"
fi
# the converter imports sentencepiece before it checks for a SentencePiece model; ours is BPE (GPT-2), so it only
# needs the import to succeed and then takes its GPT-2 path
stubs="$cache/.stubs"
mkdir -p "$stubs"
echo 'class SentencePieceProcessor: pass' > "$stubs/sentencepiece.py"

mkdir -p "$root/exports/gguf"
out="$root/exports/gguf/$name"
PYTHONPATH="$cache/gguf-py:$stubs" "$python" "$cache/convert_hf_to_gguf.py" "$export_dir" --outtype bf16 --outfile "$out-bf16.gguf"
llama-quantize "$out-bf16.gguf" "$out-Q8_0.gguf" Q8_0
ls -la "$out"-*.gguf
