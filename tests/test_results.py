# J5: infra/results.py, the results store, on fake model folders in a temporary directory (plain Python, no torch).
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infra import results  # noqa: E402

BENCH = {"hellaswag": {"acc": 0.29, "acc_norm": 0.32, "n": 10042}, "lambada": {"acc": 0.20, "ppl": 485.6},
         "arc_easy": {"acc": 0.47}, "piqa": {"acc": 0.62}, "winogrande": {"acc": 0.50}}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "STORE", str(tmp_path / "results.jsonl"))
    monkeypatch.setattr(results, "DOC", str(tmp_path / "results.md"))
    return tmp_path


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def library_model(root, mid="base-x", when="2026-11-15 16:51"):
    d = os.path.join(root, mid)
    write(os.path.join(d, "meta.json"), {"id": mid, "name": "Base X", "params": 123, "model_type": "autoregressive"})
    write(os.path.join(d, "benchmarks.json"), {"results": BENCH, "context": 512, "measured_at": when})
    write(os.path.join(d, "eval.json"), {"evaluated_at": when, "heldout": {"wikitext": {"loss": 4.2, "perplexity": 66.7,
                                                                                       "title": "WikiText-2"}},
                                         "generation": {"greedy_repeated_4grams": 0.3, "settings": {"tokens": 100}}})
    return d


def test_scan_reads_a_library_folder_once(store):
    d = library_model(str(store / "models"))
    assert results.scan([d]) == 7                  # five tasks, one held-out set, one generation record
    assert results.scan([d]) == 0                  # the same files again: nothing new
    recs = results.load()
    b = [r for r in recs if r["kind"] == "benchmark" and r["key"] == "hellaswag"][0]
    assert (b["model"], b["name"], b["numbers"]["acc_norm"], b["settings"]["context"]) == ("base-x", "Base X", 0.32, 512)
    assert "recorded_at" in b and "commit" in b


def test_latest_measurement_wins(store):
    d = library_model(str(store / "models"))
    results.scan([d])
    write(os.path.join(d, "benchmarks.json"), {"results": {"piqa": {"acc": 0.65}}, "measured_at": "2026-11-17 09:00"})
    assert results.record_file(os.path.join(d, "benchmarks.json"), "base-x") == 1
    piqa = results.latest(results.load(), "benchmark", "base-x")[("benchmark", "base-x", "piqa")]
    assert piqa["numbers"]["acc"] == 0.65
    assert len([r for r in results.load() if r["key"] == "piqa"]) == 2    # the earlier one is kept


def test_diffusion_run_and_chat_files(store):
    run = str(store / "runs" / "f6")
    write(os.path.join(run, "metrics.json"), {"model_type": "masked-diffusion", "params": 124_000_000, "tokens": 999_424_000,
                                              "best_bound": 3.9231, "finished_at": "2026-11-16 20:00",
                                              "datasets": ["BaoMinhTran/mini-fineweb-edu@full"]})
    write(os.path.join(run, "quality.json"), {"measured_at": "2026-11-16 21:00", "results": {
        "real continuation": {"summary": {"judge_loss": 3.06, "repeated_4grams": 0.004}}}})
    open(os.path.join(run, "model.pt"), "w").close()
    write(str(store / "outputs" / "dpo" / "chat_eval.json"), {"measured_at": "2026-11-15 17:46", "results": {
        "chat-1": {"answer_loss": 1.4, "greedy_repeated_4grams": 0.22, "ended_by_itself": "2/12"}}})
    assert results.scan([run, str(store / "outputs" / "dpo")]) == 3
    recs = {r["kind"]: r for r in results.load()}
    assert recs["bound"]["model"] == "diffusion:f6"
    assert recs["bound"]["name"] == "Diffusion 124M · FineWeb-Edu full + Cosmopedia · 999M tokens"
    assert recs["chat"]["numbers"]["ended"] == round(2 / 12, 3)
    page = results.render()
    assert "| Diffusion 124M · FineWeb-Edu full + Cosmopedia · 999M tokens | 3.923 | 999M | real continuation |" in page


def test_render_benchmarks_with_average(store):
    results.scan([library_model(str(store / "models"))])
    page = results.render()
    assert "| Base X | 32.0 | 20.0 | 47.0 | 62.0 | 50.0 | 42.2 | 2026-11-15 |" in page
    assert "| *OpenAI GPT-2 small (published)* |" in page


def test_record_file_never_raises(store):
    assert results.record_file(str(store / "missing.json"), "x") == 0
    (store / "bad.json").write_text("{not json")
    assert results.record_file(str(store / "bad.json"), "x") == 0


def test_results_page_is_up_to_date():
    """docs/results.md is generated from results/results.jsonl: regenerate it when the store changes."""
    assert results.write_doc(check=True), "run: python3 -m infra.results render"
