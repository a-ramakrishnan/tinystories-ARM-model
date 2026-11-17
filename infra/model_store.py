# J4: move models between Colab and the Mac through a private Hugging Face Hub repository, instead of Drive
# downloads, zips and browser clicks.
#
#   On Colab (after training):   python3 -m infra.model_store push diffusion /content/drive/MyDrive/diffusion-runs/f6-124m-stage2 --name f6-124m-stage2-2p5b
#   On the Mac:                  python3 -m infra.model_store pull diffusion f6-124m-stage2-2p5b
#                                python3 -m infra.model_store list
#
# Kinds and where `pull` puts them:
#   library    a model library folder (best_model_params.pt, model_config.json, meta.json, ...)  -> models/<name>
#   diffusion  a diffusion run folder (model.pt, config.json, metrics.json, loss.png, ...)       -> diffusion_models/runs/<name>
#   run        any other folder (e.g. a chat model's outputs)                                     -> outputs/<name>
#
# Every push writes manifest.json (size and SHA-256 of each file) next to the files; pull checks every file against
# it and refuses to overwrite a different local folder unless --force. Training state (state.pt: weights plus the
# optimizer, 3x the size) and snapshots are left out unless --with-state.
#
# Authentication is Hugging Face's own: an access token with write permission, as the HF_TOKEN environment variable
# or from `hf auth login` (on Colab: a notebook secret named HF_TOKEN). The repository is private;
# HF_MODEL_REPO overrides its name (default: <your username>/tinystories-arm-models).
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KINDS = {"library": os.path.join(ROOT, "models"), "diffusion": os.path.join(ROOT, "diffusion_models", "runs"),
         "run": os.path.join(ROOT, "outputs")}
MANIFEST = "manifest.json"
STATE_FILES = ("state.pt", "state.pt.tmp")
SKIP_DIRS = ("snapshots", "__pycache__", ".ipynb_checkpoints")
REPO_NAME = "tinystories-arm-models"


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def files_to_send(folder, with_state=False):
    """Relative paths of the files a push sends (no state, snapshots or caches unless with_state)."""
    out = []
    for d, dirs, files in os.walk(folder):
        dirs[:] = sorted(x for x in dirs if with_state or x not in SKIP_DIRS)
        for f in sorted(files):
            if f == MANIFEST or f.startswith(".") or (not with_state and f in STATE_FILES):
                continue
            out.append(os.path.relpath(os.path.join(d, f), folder))
    return out


def make_manifest(folder, kind, name, with_state=False):
    files = {p: {"bytes": os.path.getsize(os.path.join(folder, p)), "sha256": sha256(os.path.join(folder, p))}
             for p in files_to_send(folder, with_state)}
    return {"kind": kind, "name": name, "files": files, "pushed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "total_bytes": sum(f["bytes"] for f in files.values())}


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


def repo_id(api=None):
    if os.environ.get("HF_MODEL_REPO"):
        return os.environ["HF_MODEL_REPO"]
    from huggingface_hub import HfApi
    api = api or HfApi()
    return f"{api.whoami()['name']}/{REPO_NAME}"


def _check(kind):
    if kind not in KINDS:
        raise SystemExit(f"kind must be one of {', '.join(KINDS)}")


def push(kind, folder, name=None, with_state=False, api=None, say=print):
    """Upload a folder (plus its manifest) to <repo>/<kind>/<name>/."""
    _check(kind)
    from huggingface_hub import HfApi
    api = api or HfApi()
    folder = os.path.abspath(folder)
    name = name or os.path.basename(folder.rstrip("/"))
    repo = repo_id(api)
    man = make_manifest(folder, kind, name, with_state)
    if not man["files"]:
        raise SystemExit(f"nothing to push in {folder}")
    with open(os.path.join(folder, MANIFEST), "w") as f:
        json.dump(man, f, indent=1)
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    say(f"Pushing {len(man['files'])} files ({man['total_bytes'] / 1e6:,.0f} MB) to {repo}/{kind}/{name} …")
    api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=f"{kind}/{name}",
                      allow_patterns=list(man["files"]) + [MANIFEST],
                      commit_message=f"{kind}/{name}: {len(man['files'])} files, {man['total_bytes'] / 1e6:,.0f} MB")
    say("Done.")
    return man


def pull(kind, name, dest=None, force=False, api=None, say=print):
    """Download <repo>/<kind>/<name>/ to its local place, check every file, and move it into place."""
    _check(kind)
    from huggingface_hub import HfApi, snapshot_download
    api = api or HfApi()
    repo = repo_id(api)
    dest = dest or os.path.join(KINDS[kind], name)
    cache = snapshot_download(repo, repo_type="model", allow_patterns=[f"{kind}/{name}/*"])
    src = os.path.join(cache, kind, name)
    if not os.path.exists(os.path.join(src, MANIFEST)):
        raise SystemExit(f"{repo} has no {kind}/{name} (see `python3 -m infra.model_store list`)")
    man = json.load(open(os.path.join(src, MANIFEST)))
    bad = verify(src, man)
    if bad:
        raise SystemExit("download doesn't match its manifest: " + "; ".join(bad))
    if os.path.exists(dest) and not force:
        if os.path.exists(os.path.join(dest, MANIFEST)) and not verify(dest, man):
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
    return dest


def list_remote(api=None):
    """{kind: [names]} in the repository."""
    from huggingface_hub import HfApi
    api = api or HfApi()
    out = {}
    for path in api.list_repo_files(repo_id(api), repo_type="model"):
        parts = path.split("/")
        if len(parts) >= 3 and parts[-1] == MANIFEST and parts[0] in KINDS:
            out.setdefault(parts[0], []).append("/".join(parts[1:-1]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Move models through a private Hugging Face Hub repository.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push", help="upload a folder")
    p.add_argument("kind", choices=list(KINDS))
    p.add_argument("folder")
    p.add_argument("--name", default=None, help="name in the repository (default: the folder's name)")
    p.add_argument("--with-state", action="store_true", help="also send state.pt and snapshots/")
    q = sub.add_parser("pull", help="download into models/, diffusion_models/runs/ or outputs/")
    q.add_argument("kind", choices=list(KINDS))
    q.add_argument("name")
    q.add_argument("--dest", default=None)
    q.add_argument("--force", action="store_true")
    sub.add_parser("list", help="what the repository holds")
    v = sub.add_parser("verify", help="check a local folder against its manifest.json")
    v.add_argument("folder")
    a = ap.parse_args(argv)
    if a.cmd == "push":
        push(a.kind, a.folder, a.name, a.with_state)
    elif a.cmd == "pull":
        pull(a.kind, a.name, a.dest, a.force)
    elif a.cmd == "list":
        for kind, names in sorted(list_remote().items()):
            print(f"{kind}: " + ", ".join(sorted(names)))
    elif a.cmd == "verify":
        bad = verify(a.folder)
        print("\n".join(bad) if bad else "all files match manifest.json")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
