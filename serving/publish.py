# Publish a served copy (plan S6): one public Hugging Face repository per model with both formats, the Qwen3
# safetensors from serving/export_hf.py (for transformers) and the GGUF files from serving/gguf.sh (llama.cpp, Ollama,
# LM Studio pick one by its quant tag), with the card from docs/model_cards/<repo>.md.
#
#   python3 -m serving.publish <export folder> <gguf name> <repo>
#   python3 -m serving.publish exports/chat-mini-fineweb-ed-202611152150-2746ba-qwen3 fineweb-edu-124m-chat-dpo \
#       aramkris/fineweb-edu-124m-chat-dpo-qwen3
import argparse
import os
import shutil
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stage(export_dir, gguf_name, repo, into):
    for f in os.listdir(export_dir):
        shutil.copy(os.path.join(export_dir, f), into)
    gg = os.path.join(ROOT, "exports", "gguf")
    files = sorted(f for f in os.listdir(gg) if f.startswith(gguf_name + "-") and f.endswith(".gguf"))
    if not files:
        raise SystemExit(f"No GGUF files {gguf_name}-*.gguf in exports/gguf (run serving/gguf.sh)")
    for f in files:
        os.link(os.path.join(gg, f), os.path.join(into, f))       # a hard link: no second copy of 500 MB
    card = os.path.join(ROOT, "docs", "model_cards", repo.split("/")[-1] + ".md")
    if not os.path.exists(card):
        raise SystemExit(f"No model card {os.path.relpath(card, ROOT)}")
    shutil.copy(card, os.path.join(into, "README.md"))
    return sorted(os.listdir(into))


def main():
    ap = argparse.ArgumentParser(description="Publish an exported model (Qwen3 safetensors + GGUF) to Hugging Face.")
    ap.add_argument("export_dir")
    ap.add_argument("gguf_name")
    ap.add_argument("repo")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "exports")) as tmp:
        files = stage(a.export_dir, a.gguf_name, a.repo, tmp)
        print(f"{a.repo}: {', '.join(files)}")
        api.create_repo(a.repo, private=a.private, exist_ok=True)
        api.upload_folder(folder_path=tmp, repo_id=a.repo, commit_message="Qwen3-format copy and GGUF files (serving/)")
    print(f"https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
