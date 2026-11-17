# J4: infra/model_store.py (one Hub repository per model) against a fake Hub kept in a temporary folder: no network,
# no token.
import os
import shutil
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra import model_store as ms  # noqa: E402


class FakeHub:
    """Just enough of HfApi: a repository is a folder under root/<user>/<name>."""

    def __init__(self, root):
        self.root = root
        self.created = []

    def whoami(self):
        return {"name": "someone"}

    def create_repo(self, repo, repo_type, private, exist_ok):
        self.created.append((repo, private))
        os.makedirs(os.path.join(self.root, repo), exist_ok=True)

    def upload_folder(self, repo_id, folder_path, commit_message, allow_patterns=None):
        names = allow_patterns or [os.path.relpath(os.path.join(d, f), folder_path)
                                   for d, _, fs in os.walk(folder_path) for f in fs]
        for p in names:
            dst = os.path.join(self.root, repo_id, p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(folder_path, p), dst)

    def list_models(self, author, filter):
        out = []
        base = os.path.join(self.root, author)
        for name in sorted(os.listdir(base)) if os.path.exists(base) else []:
            card = os.path.join(base, name, "README.md")
            if os.path.exists(card) and filter in open(card).read():
                out.append(SimpleNamespace(id=f"{author}/{name}"))
        return out


@pytest.fixture
def hub(tmp_path, monkeypatch):
    h = FakeHub(str(tmp_path / "hub"))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: h)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, repo_type: os.path.join(h.root, repo))
    return h


def make_run(folder, meta_name="Diffusion 124M"):
    os.makedirs(os.path.join(folder, "snapshots"))
    files = {"model.pt": b"weights" * 1000, "config.json": b'{"name": "%s"}' % meta_name.encode(),
             "metrics.json": b'{"best": 3.9}', "state.pt": b"optimizer" * 3000, "snapshots/state-update010000.pt": b"old"}
    for name, data in files.items():
        with open(os.path.join(folder, name), "wb") as f:
            f.write(data)


def quiet(*_):
    pass


def test_push_then_pull_round_trip(tmp_path, hub):
    src = str(tmp_path / "f6-124m-stage2")
    make_run(src)
    man = ms.push("diffusion", src, "tinystories-arm-124m-diffusion", say=quiet)
    repo_dir = os.path.join(hub.root, "someone", "tinystories-arm-124m-diffusion")
    assert hub.created == [("someone/tinystories-arm-124m-diffusion", False)]   # public, under the user's name
    assert set(man["files"]) == {"model.pt", "config.json", "metrics.json"}      # no state.pt, no snapshots
    assert sorted(os.listdir(repo_dir)) == ["README.md", "config.json", "manifest.json", "metrics.json", "model.pt"]
    assert "Diffusion 124M" in open(os.path.join(repo_dir, "README.md")).read()  # card from the folder's metadata
    assert not os.path.exists(os.path.join(src, "manifest.json"))                  # the run folder is left alone
    dest = str(tmp_path / "local")
    ms.pull("tinystories-arm-124m-diffusion", dest=dest, say=quiet)
    assert open(os.path.join(dest, "model.pt"), "rb").read() == b"weights" * 1000
    assert ms.verify(dest) == []


def test_pull_puts_it_where_the_kind_says(tmp_path, hub, monkeypatch):
    monkeypatch.setitem(ms.KINDS, "library", str(tmp_path / "models"))
    src = str(tmp_path / "mini-fineweb-edu-ful-x")
    make_run(src)
    ms.push("library", src, "tinystories-arm-124m-base", say=quiet)
    assert ms.pull("tinystories-arm-124m-base", say=quiet) == str(tmp_path / "models" / "mini-fineweb-edu-ful-x")


def test_custom_card_and_list(tmp_path, hub):
    card = tmp_path / "card.md"
    card.write_text("---\ntags:\n  - tinystories-arm\n---\n# The base model\n")
    for repo in ("tinystories-arm-124m-base", "tinystories-arm-124m-chat-dpo"):
        src = str(tmp_path / repo)
        make_run(src)
        ms.push("library", src, repo, card=str(card), say=quiet)
    assert open(os.path.join(hub.root, "someone", "tinystories-arm-124m-base", "README.md")).read().startswith("---")
    assert ms.list_remote() == ["someone/tinystories-arm-124m-base", "someone/tinystories-arm-124m-chat-dpo"]


def test_with_state_sends_everything(tmp_path):
    src = str(tmp_path / "f6")
    make_run(src)
    assert "state.pt" in ms.files_to_send(src, with_state=True)
    assert os.path.join("snapshots", "state-update010000.pt") in ms.files_to_send(src, with_state=True)


def test_pull_detects_a_corrupt_download(tmp_path, hub):
    src = str(tmp_path / "f6")
    make_run(src)
    ms.push("diffusion", src, "x", say=quiet)
    with open(os.path.join(hub.root, "someone", "x", "model.pt"), "ab") as f:
        f.write(b"!")
    with pytest.raises(SystemExit, match="wrong size"):
        ms.pull("x", dest=str(tmp_path / "out"), say=quiet)


def test_pull_wont_overwrite_a_different_folder(tmp_path, hub):
    src = str(tmp_path / "f6")
    make_run(src)
    ms.push("diffusion", src, "x", say=quiet)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "model.pt").write_bytes(b"something else")
    with pytest.raises(SystemExit, match="--force"):
        ms.pull("x", dest=str(dest), say=quiet)
    ms.pull("x", dest=str(dest), force=True, say=quiet)
    assert ms.verify(str(dest)) == []
    msgs = []
    ms.pull("x", dest=str(dest), say=msgs.append)            # a second pull: nothing to do
    assert "up to date" in msgs[0]


def test_full_repo_names():
    api = SimpleNamespace(whoami=lambda: {"name": "me"})
    assert ms.full_repo("model-a", api) == "me/model-a"
    assert ms.full_repo("other/model-b", api) == "other/model-b"
