"""Tests for src/phase5/audit.py."""
import pytest

from src.phase5 import audit as A


def claims():
    out = []
    for g in ("a", "b"):
        for i in range(1000):
            out.append({"generator": g, "image_id": i, "category": "dog", "caption": "a dog",
                        "hall": i % 2 == 0, "score": float(i % 100)})
    return out


def test_sample_strata_sizes_and_determinism():
    tau = {"a": 50.0, "b": 50.0}
    r1, r2 = A.sample(claims(), tau), A.sample(claims(), tau)
    assert [r["audit_id"] for r in r1] == [r["audit_id"] for r in r2]
    assert [(r["generator"], r["image_id"]) for r in r1] == [(r["generator"], r["image_id"]) for r in r2]
    assert len(r1) == 400
    for g in ("a", "b"):
        rs = [r for r in r1 if r["generator"] == g]
        assert sum(r["stratum"] == "hall_above" and r["hall"] and r["score"] > 50 for r in rs) == 100
        assert sum(r["stratum"] == "hall_below" and r["hall"] and r["score"] <= 50 for r in rs) == 50
        assert sum(r["stratum"] == "supported" and not r["hall"] for r in rs) == 50


def test_sample_refuses_small_pool():
    with pytest.raises(ValueError):
        A.sample(claims(), {"a": 1e9, "b": 1e9})


def test_annotator_file_is_blind(tmp_path):
    rows = A.sample(claims(), {"a": 50.0, "b": 50.0})
    A.write_files(rows, tmp_path, lambda i: f"https://example.org/{i}.jpg")
    head = (tmp_path / "annotator_tasks.csv").read_text().splitlines()[0]
    assert "score" not in head and "generator" not in head and "hallucinated" not in head


def test_cohen_kappa():
    assert A.cohen_kappa(["present", "absent"] * 5, ["present", "absent"] * 5) == pytest.approx(1.0)
    a = ["present"] * 6 + ["absent"] * 4
    b = ["present"] * 4 + ["absent"] * 2 + ["absent"] * 4
    # po = 0.8, pe = 0.6*0.4 + 0.4*0.6 = 0.48 -> kappa = 0.32/0.52
    assert A.cohen_kappa(a, b) == pytest.approx(0.32 / 0.52)


def test_noise_rates():
    key = [{"audit_id": "x1", "stratum": "hall_above"}, {"audit_id": "x2", "stratum": "hall_above"},
           {"audit_id": "x3", "stratum": "supported"}]
    r = A.noise_rates(key, {"x1": "present", "x2": "absent", "x3": "absent"})
    assert r["hall_above"]["chair_wrong"] == 0.5 and r["supported"]["chair_wrong"] == 1.0
