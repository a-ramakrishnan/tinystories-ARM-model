# J4: move models between Colab and the Mac through Hugging Face Hub, one public repository per model, instead of
# Drive downloads, zips and browser clicks.
#
#   python3 -m infra.model_store push library models/<id> tinystories-arm-124m-base --card card.md
#   python3 -m infra.model_store push diffusion /content/drive/MyDrive/diffusion-runs/f6-124m-stage2 tinystories-arm-124m-diffusion
#   python3 -m infra.model_store pull tinystories-arm-124m-base        # -> models/<id>, checked
#   python3 -m infra.model_store list                                   # every repository tagged tinystories-arm
#
# Kinds and where `pull` puts them (the local folder name travels in manifest.json):
#   library    a model library folder (best_model_params.pt, model_config.json, meta.json, ...)  -> models/<id>
#   diffusion  a diffusion run folder (model.pt, config.json, metrics.json, loss.png, ...)       -> diffusion_models/runs/<name>
#   run        any other folder                                                                   -> outputs/<name>
#
# Every push writes manifest.json (kind, local folder name, size and SHA-256 of each file) at the top of the
# repository; pull checks every file against it and refuses to overwrite a different local folder unless --force.
# Training state (state.pt: weights plus optimizer, 3x the size) and snapshots stay behind unless --with-state.
# The model card is --card's file, or else a short one written from the folder's metadata; it carries the
# `tinystories-arm` tag that `list` looks for.
#
# Authentication is Hugging Face's own: a token with write permission, as HF_TOKEN or from `hf auth login` (on Colab:
# the notebook secret HF_TOKEN). New repositories are public (the models hold nothing private); --private otherwise.
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = {"library": os.path.join(ROOT, "models"), "diffusion": os.path.join(ROOT, "diffusion_models", "runs"),
         "run": os.path.join(ROOT, "outputs")}
MANIFEST = "manifest.json"
CARD = "README.md"
TAG = "tinystories-arm"
STATE_FILES = ("state.pt", "state.pt.tmp")
SKIP_DIRS = ("snapshots", "__pycache__", ".ipynb_checkpoints")
GITHUB = "https://github.com/a-ramakrishnan/tinystories-ARM-model"


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def files_to_send(folder, with_state=False):
    """Relative paths of the files a push sends (no state, snapshots, caches or an old card unless with_state)."""
    out = []
    for d, dirs, files in os.walk(folder):
        dirs[:] = sorted(x for x in dirs if with_state or x not in SKIP_DIRS)
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(d, f), folder)
            if rel in (MANIFEST, CARD) or f.startswith(".") or (not with_state and f in STATE_FILES):
                continue
            out.append(rel)
    return out


def make_manifest(folder, kind, with_state=False):
    files = {p: {"bytes": os.path.getsize(os.path.join(folder, p)), "sha256": sha256(os.path.join(folder, p))}
             for p in files_to_send(folder, with_state)}
    return {"kind": kind, "local_name": os.path.basename(os.path.abspath(folder).rstrip("/")), "files": files,
            "pushed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "total_bytes": sum(f["bytes"] for f in files.values())}


def verify(folder, manifest=None):
    """List of problems (missing, wrong size, wrong checksum); empty means the folder matches its manifest."""
    manifest = manifest or json.load(open(os.path.join(folder, MANIFEST)))
    problems = []
    for p, info in manifest["files"].items():
        f = os.path.join(folder, p)
        if not os.path.exists(f):
            problems.append(f"missing: {p}")
        elif os.path.getsize(f) != info["bytes"]:
            problems.append(f"wrong size: {p}")
        elif sha256(f) != info["sha256"]:
            problems.append(f"wrong checksum: {p}")
    return problems


def full_repo(name, api):
    """'name' -> '<username>/name'; 'someone/name' stays as it is."""
    return name if "/" in name else f"{api.whoami()['name']}/{name}"


def default_card(folder, kind, repo):
    """A short model card from the folder's own metadata, for pushes without --card."""
    meta = {}
    for f in ("meta.json", "config.json", "model_config.json"):
        p = os.path.join(folder, f)
        if os.path.exists(p):
            meta = json.load(open(p))
            break
    title = meta.get("name") or os.path.basename(folder.rstrip("/"))
    what = "masked diffusion language model" if kind == "diffusion" else "language model"
    return (f"---\nlicense: mit\nlanguage: en\nlibrary_name: pytorch\ntags:\n  - {TAG}\n  - from-scratch\n---\n\n"
            f"# {title}\n\nA small {what} trained from scratch in [tinystories-ARM-model]({GITHUB}).\n\n"
            f"```bash\ngit clone {GITHUB} && cd tinystories-ARM-model\npython3 -m infra.model_store pull {repo}\n```\n")


def push(kind, folder, repo, card=None, with_state=False, api=None, say=print, private=False):
    """Upload a folder to its own repository: the files, manifest.json, and a model card (README.md)."""
    if kind not in KINDS:
        raise SystemExit(f"kind must be one of {', '.join(KINDS)}")
    from huggingface_hub import HfApi
    api = api or HfApi()
    folder = os.path.abspath(folder)
    repo = full_repo(repo, api)
    man = make_manifest(folder, kind, with_state)
    if not man["files"]:
        raise SystemExit(f"nothing to push in {folder}")
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    say(f"Pushing {len(man['files'])} files ({man['total_bytes'] / 1e6:,.0f} MB) to {repo} …")
    with tempfile.TemporaryDirectory() as extra:      # manifest and card go up without touching the run folder
        with open(os.path.join(extra, MANIFEST), "w") as f:
            json.dump(man, f, indent=1)
        text = open(card).read() if card else default_card(folder, kind, repo)
        if TAG not in text:
            say(f"note: the card has no '{TAG}' tag, so `list` won't show this repository")
        with open(os.path.join(extra, CARD), "w") as f:
            f.write(text)
        api.upload_folder(repo_id=repo, folder_path=folder, allow_patterns=list(man["files"]),
                          commit_message=f"{man['local_name']}: {len(man['files'])} files, {man['total_bytes'] / 1e6:,.0f} MB")
        api.upload_folder(repo_id=repo, folder_path=extra, commit_message="manifest and model card")
    say(f"Done: https://huggingface.co/{repo}")
    return man


def pull(repo, dest=None, force=False, api=None, say=print):
    """Download a repository to its local place (from its manifest), check every file, and move it into place."""
    from huggingface_hub import HfApi, snapshot_download
    api = api or HfApi()
    repo = full_repo(repo, api)
    src = snapshot_download(repo, repo_type="model")
    if not os.path.exists(os.path.join(src, MANIFEST)):
        raise SystemExit(f"{repo} has no {MANIFEST}: it wasn't pushed with this tool")
    man = json.load(open(os.path.join(src, MANIFEST)))
    bad = verify(src, man)
    if bad:
        raise SystemExit("download doesn't match its manifest: " + "; ".join(bad))
    dest = dest or os.path.join(KINDS[man["kind"]], man["local_name"])
    if os.path.exists(dest) and not force:
        if not verify(dest, man):
            say(f"{dest} is already up to date.")
            return dest
        if os.listdir(dest):
            raise SystemExit(f"{dest} exists and differs; pass --force to replace the files it shares with the download")
    os.makedirs(dest, exist_ok=True)
    for p in list(man["files"]) + [MANIFEST]:
        os.makedirs(os.path.dirname(os.path.join(dest, p)), exist_ok=True)
        tmp = os.path.join(dest, p + ".part")
        shutil.copyfile(os.path.join(src, p), tmp)      # the cache holds links; copy real files
        os.replace(tmp, os.path.join(dest, p))
    say(f"Pulled {len(man['files'])} files ({man['total_bytes'] / 1e6:,.0f} MB) into {dest}; checksums match.")
    try:                                                # J5: its evaluation results go into the results store
        from infra import results
        n = results.scan([dest])
        if n:
            results.write_doc()
            say(f"results store: {n} new records")
    except Exception as e:
        say(f"results store: not updated ({e})")
    return dest


def list_remote(api=None, author=None):
    """Repository ids tagged tinystories-arm (yours, or author's)."""
    from huggingface_hub import HfApi
    api = api or HfApi()
    author = author or api.whoami()["name"]
    return sorted(m.id for m in api.list_models(author=author, filter=TAG))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Move models through Hugging Face Hub, one repository per model.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="upload a folder to its own repository")
    p.add_argument("kind", choices=list(KINDS))
    p.add_argument("folder")
    p.add_argument("repo", help="repository name (under your account) or user/name")
    p.add_argument("--card", default=None, help="model card (Markdown) to use as README.md")
    p.add_argument("--with-state", action="store_true", help="also send state.pt and snapshots/")
    p.add_argument("--private", action="store_true", help="create the repository private (if it doesn't exist yet)")
    q = sub.add_parser("pull", help="download into models/, diffusion_models/runs/ or outputs/")
    q.add_argument("repo")
    q.add_argument("--dest", default=None)
    q.add_argument("--force", action="store_true")
    li = sub.add_parser("list", help="your repositories tagged tinystories-arm")
    li.add_argument("--author", default=None)
    v = sub.add_parser("verify", help="check a local folder against its manifest.json")
    v.add_argument("folder")
    a = ap.parse_args(argv)
    if a.cmd == "push":
        push(a.kind, a.folder, a.repo, a.card, a.with_state, private=a.private)
    elif a.cmd == "pull":
        pull(a.repo, a.dest, a.force)
    elif a.cmd == "list":
        print("\n".join(list_remote(author=a.author)))
    elif a.cmd == "verify":
        bad = verify(a.folder)
        print("\n".join(bad) if bad else "all files match manifest.json")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
