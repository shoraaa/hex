from pathlib import Path

from hexbench import cli


def test_time_benchmark_forwards_seed_profile(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_time_benchmark(*args, **kwargs):
        captured.update(kwargs)
        return {"map_count": 1, "time_limits_ms": [500]}

    monkeypatch.setattr(cli, "lns_time_benchmark", fake_time_benchmark)
    cli.main(
        [
            "lns-time-benchmark",
            "--method",
            "alns",
            "--time-limits-ms",
            "500",
            "--seed-profile",
            "reduced_no_local",
            "--report",
            str(tmp_path),
        ]
    )

    assert captured["seed_profile"] == "reduced_no_local"
