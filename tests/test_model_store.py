# J4: infra/model_store.py against a fake Hugging Face Hub kept in a temporary folder (no network, no token).
import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra import model_store as ms  # noqa: E402


class FakeHub:
    """Just enough of HfApi: a repository is a folder."""

    def __init__(self, root):
        self.root = root
        self.created = []

    def whoami(self):
        return {"name": "someone"}

    def create_repo(self, repo, repo_type, private, exist_ok):
        self.created.append((repo, private))

    def upload_folder(self, repo_id, folder_path, path_in_repo, allow_patterns, commit_message):
        for p in allow_patterns:
            dst = os.path.join(self.root, path_in_repo, p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(folder_path, p), dst)

    def list_repo_files(self, repo, repo_type):
        return [os.path.relpath(os.path.join(d, f), self.root) for d, _, fs in os.walk(self.root) for f in fs]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    h = FakeHub(str(tmp_path / "hub"))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: h)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, repo_type, allow_patterns: h.root)
    monkeypatch.delenv("HF_MODEL_REPO", raising=False)
    return h


def make_run(folder):
    os.makedirs(os.path.join(folder, "snapshots"))
    for name, data in {"model.pt": b"weights" * 1000, "config.json": b"{}", "metrics.json": b'{"best": 3.9}',
                       "state.pt": b"optimizer" * 3000, "snapshots/state-update010000.pt": b"old"}.items():
        with open(os.path.join(folder, name), "wb") as f:
            f.write(data)


def test_push_then_pull_round_trip(tmp_path, hub):
    src = str(tmp_path / "f6")
    make_run(src)
    man = ms.push("diffusion", src, name="f6-2p5b", say=lambda *_: None)
    assert hub.created == [("someone/tinystories-arm-models", True)]           # private, under the user's name
    assert set(man["files"]) == {"model.pt", "config.json", "metrics.json"}     # no state.pt, no snapshots
    dest = str(tmp_path / "local" / "f6-2p5b")
    ms.pull("diffusion", "f6-2p5b", dest=dest, say=lambda *_: None)
    assert open(os.path.join(dest, "model.pt"), "rb").read() == b"weights" * 1000
    assert ms.verify(dest) == []
    assert ms.list_remote() == {"diffusion": ["f6-2p5b"]}


def test_with_state_sends_everything(tmp_path):
    src = str(tmp_path / "f6")
    make_run(src)
    assert "state.pt" in ms.files_to_send(src, with_state=True)
    assert os.path.join("snapshots", "state-update010000.pt") in ms.files_to_send(src, with_state=True)


def test_pull_detects_a_corrupt_download(tmp_path, hub):
    src = str(tmp_path / "f6")
    make_run(src)
    ms.push("diffusion", src, name="x", say=lambda *_: None)
    with open(os.path.join(hub.root, "diffusion", "x", "model.pt"), "ab") as f:
        f.write(b"!")
    with pytest.raises(SystemExit, match="wrong size"):
        ms.pull("diffusion", "x", dest=str(tmp_path / "out"), say=lambda *_: None)


def test_pull_wont_overwrite_a_different_folder(tmp_path, hub):
    src = str(tmp_path / "f6")
    make_run(src)
    ms.push("diffusion", src, name="x", say=lambda *_: None)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "model.pt").write_bytes(b"something else")
    with pytest.raises(SystemExit, match="--force"):
        ms.pull("diffusion", "x", dest=str(dest), say=lambda *_: None)
    ms.pull("diffusion", "x", dest=str(dest), force=True, say=lambda *_: None)
    assert ms.verify(str(dest)) == []
    msgs = []
    ms.pull("diffusion", "x", dest=str(dest), say=msgs.append)           # a second pull: nothing to do
    assert "up to date" in msgs[0]


def test_repo_name_can_be_overridden(monkeypatch):
    monkeypatch.setenv("HF_MODEL_REPO", "me/elsewhere")
    assert ms.repo_id() == "me/elsewhere"
