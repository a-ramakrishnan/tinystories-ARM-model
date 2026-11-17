# J6: the evaluation queue (infra/eval_queue.py): resuming, failures, out-of-memory retries and the batch choice,
# with stand-in tasks (no model, no GPU).
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from infra import eval_queue as Q  # noqa: E402


def target(tmp_path):
    return {"kind": "diffusion", "name": "run", "dir": str(tmp_path)}


def test_done_tasks_are_skipped_next_time(tmp_path):
    calls = []
    fns = {n: (lambda t, ctx, n=n: calls.append(n) or f"{n} ok") for n in ("a", "b")}
    t = target(tmp_path)
    assert Q.run_target(t, ["a", "b"], "cpu", fns=fns, say=lambda m: None) == {"a": "done", "b": "done"}
    assert Q.run_target(t, ["a", "b"], "cpu", fns=fns, say=lambda m: None) == {"a": "skipped", "b": "skipped"}
    assert calls == ["a", "b"]
    state = json.load(open(tmp_path / Q.STATE))
    assert state["tasks"]["a"]["status"] == "done" and state["tasks"]["a"]["summary"] == "a ok"
    Q.run_target(t, ["a"], "cpu", fns=fns, redo=True, say=lambda m: None)
    assert calls == ["a", "b", "a"]


def test_a_failure_is_recorded_and_the_rest_still_runs(tmp_path):
    def boom(t, ctx):
        raise ValueError("no data")
    fns = {"a": boom, "b": lambda t, ctx: "fine"}
    t = target(tmp_path)
    assert Q.run_target(t, ["a", "b"], "cpu", fns=fns, say=lambda m: None) == {"a": "failed", "b": "done"}
    state = json.load(open(tmp_path / Q.STATE))
    assert state["tasks"]["a"]["status"] == "failed" and "no data" in state["tasks"]["a"]["summary"]
    # a failed task is tried again on the next run
    fns["a"] = lambda t, ctx: "fixed"
    assert Q.run_target(t, ["a", "b"], "cpu", fns=fns, say=lambda m: None) == {"a": "done", "b": "skipped"}


def test_out_of_memory_halves_the_batch(tmp_path):
    seen = []

    def task(t, ctx):
        ctx["last_batch"] = ctx["batch"] or 32
        seen.append(ctx["last_batch"])
        if ctx["last_batch"] > 8:
            raise RuntimeError("MPS backend out of memory (MPS allocated: 11 GB)")
        return "ok"
    out = Q.run_target(target(tmp_path), ["bench:x"], "cpu", fns={"bench:x": task}, say=lambda m: None)
    assert out == {"bench:x": "done"} and seen == [32, 16, 8]
    assert json.load(open(tmp_path / Q.STATE))["tasks"]["bench:x"]["batch"] == 8


def test_quick_saves_no_state(tmp_path):
    Q.run_target(target(tmp_path), ["a"], "cpu", quick=True, fns={"a": lambda t, ctx: "ok"}, say=lambda m: None)
    assert not os.path.exists(tmp_path / Q.STATE)


def test_batch_from_free_memory():
    row = 512 * 50304 * 4 * Q.LOGIT_COPIES                    # one 512-token row's logits and temporaries
    assert Q.pick_batch(None, 512, 50304, 16) == 16             # unknown (CPU): the default
    assert Q.pick_batch(row * 40 / Q.MEMORY_SHARE, 512, 50304, 16) == 32
    assert Q.pick_batch(row * 1000, 512, 50304, 16) == Q.MAX_BATCH
    assert Q.pick_batch(row, 512, 50304, 16) == Q.MIN_BATCH


def test_task_lists():
    assert Q.task_names("ar")[0] == "suite" and Q.task_names("diffusion")[:2] == ["bound", "quality"]
    assert Q.task_names("ar")[-1] == "bench:hellaswag"           # the longest last
