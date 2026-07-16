import json
from pathlib import Path

import pytest

from hexbench import tuning


def test_parameter_grid_validates_and_deduplicates() -> None:
    candidates = tuning.parameter_grid([128, 128, 256], [32], [0, 0])
    assert [candidate.as_dict() for candidate in candidates] == [
        {"fixed_iterations": 128, "min_iterations": 32, "stagnation_iterations": 0},
        {"fixed_iterations": 256, "min_iterations": 32, "stagnation_iterations": 0},
    ]
    with pytest.raises(ValueError, match="cannot exceed"):
        tuning.parameter_grid([8], [16])


def test_tune_alns_ranks_lexicographic_scores(monkeypatch, tmp_path: Path) -> None:
    case_path = tmp_path / "case.json"
    scenario = {
        "config": {
            "daySteps": [4],
            "agents": [0],
            "spots": [{"brand": 0, "pos": 1, "stocks": 1}],
        }
    }
    case_path.write_text(json.dumps(scenario))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"suite": "test", "cases": [{"path": "case.json"}]}))
    binary = tmp_path / "hexudon"
    binary.write_bytes(b"test binary")

    def fake_run_core(command, policy, payload, **kwargs):
        budget = payload["search"]["maxIterations"]
        return {
            "score": {"distinct_types": int(budget > 128), "cumulative_daily_types": 0, "total_servings": 0},
            "invalid_days": 0,
        }

    monkeypatch.setattr(tuning, "run_core", fake_run_core)
    monkeypatch.setattr(tuning, "find_binary", lambda _: binary)
    report = tuning.tune_alns(
        manifest,
        tmp_path / "report",
        fixed_iterations=[128, 256],
        jobs=1,
    )
    assert report["best"]["parameters"]["fixed_iterations"] == 256
    assert (tmp_path / "report" / "report.json").exists()

