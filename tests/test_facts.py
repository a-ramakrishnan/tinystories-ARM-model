# The facts eval (facts.py) and SFT replay (sft.py): answers are matched with their aliases, the question set is
# well formed, and a replay batch mixes plain text (every token learned) into the conversations.
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def test_answer_matching():
    from facts import is_right
    it = {"answer": "New Delhi", "aliases": ["Delhi"]}
    assert is_right("The capital of India is New Delhi.", it)
    assert is_right("It is delhi, in the north", it)
    assert not is_right("The capital of India is Mumbai.", it)


def test_question_set():
    from facts import load_facts
    items = load_facts()
    assert len(items) == 200 and len({i["question"] for i in items}) == 200
    for i in items:
        assert len(i["wrong"]) == 3 and i["answer"] not in i["wrong"] and i["stem"] and i["question"].endswith("?")


def test_replay_rows(tmp_path):
    from sft import SFTLoader
    L = 17
    for s in ("train", "test"):
        np.save(tmp_path / f"{s}_ids.npy", np.full((20, L), 7, dtype=np.uint16))
        m = np.zeros((20, L), dtype=np.uint8)
        m[:, 10:] = 1
        np.save(tmp_path / f"{s}_mask.npy", m)
    rd = tmp_path / "replay"
    rd.mkdir()
    np.arange(1000, dtype=np.uint16).tofile(rd / "replay.bin")
    x, y = SFTLoader(str(tmp_path), "cpu", 8, replay=0.25, replay_dir=str(rd)).batch()
    assert x.shape == (8, L - 1)
    assert (y[:6, :9] == -1).all() and (y[:6, 9:] == 7).all()        # conversations: answers only
    assert (y[6:] >= 0).all() and (y[6:, 1:] - y[6:, :-1] == 1).all()  # replay: consecutive text, all learned
