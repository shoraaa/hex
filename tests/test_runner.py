from __future__ import annotations

import stat
from pathlib import Path

import pytest

from hexbench.runner import stream_core


def _fake_binary(tmp_path: Path, body: str) -> Path:
    """Write an executable stand-in for the `hexudon solve` streaming core."""
    script = tmp_path / "fake_solve"
    script.write_text("#!/usr/bin/env python3\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_stream_core_reports_each_improvement_and_returns_best(tmp_path: Path) -> None:
    body = (
        "import sys, json\n"
        "sys.stdin.read()\n"
        "rank = {'congestion_mode': 'current', 'congestion': [0, 0, -2, -3, 0, 0], "
        "'workload': [1, -1, -1], 'patrol_fuel': 7}\n"
        "print(json.dumps({'score': [1, 1, 1], 'internal_rank': rank, 'actions': [[-4]]}), flush=True)\n"
        "rank['patrol_fuel'] = 8\n"
        "print(json.dumps({'score': [2, 2, 2], 'internal_rank': rank, 'actions': [[0, -3]]}), flush=True)\n"
    )
    binary = _fake_binary(tmp_path, body)
    seen: list[dict] = []
    last = stream_core(
        "alns",
        {"config": {}},
        binary=binary,
        on_improve=seen.append,
        timeout=10,
    )
    assert [tuple(record["score"]) for record in seen] == [
        (1, 1, 1),
        (2, 2, 2),
    ]
    assert last is not None and last["score"] == [2, 2, 2]
    assert last["internal_rank"]["patrol_fuel"] == 8
    assert last["actions"] == [[0, -3]]


def test_stream_core_raises_on_core_failure(tmp_path: Path) -> None:
    body = (
        "import sys\n"
        "sys.stdin.read()\n"
        "sys.stderr.write('boom')\n"
        "sys.exit(1)\n"
    )
    binary = _fake_binary(tmp_path, body)
    with pytest.raises(RuntimeError, match="boom"):
        stream_core(
            "alns", {}, binary=binary, on_improve=lambda record: None, timeout=10
        )


def test_stream_core_stops_when_should_stop_requests_it(tmp_path: Path) -> None:
    body = (
        "import sys, json, time\n"
        "sys.stdin.read()\n"
        "for i in range(100):\n"
        "    print(json.dumps({'score': [i, i, i], 'actions': [[-4]]}), flush=True)\n"
        "    time.sleep(0.05)\n"
    )
    binary = _fake_binary(tmp_path, body)
    seen: list[dict] = []
    last = stream_core(
        "alns",
        {},
        binary=binary,
        on_improve=seen.append,
        timeout=10,
        should_stop=lambda: len(seen) >= 1,
    )
    # The caller stop is honoured after the first line without raising.
    assert len(seen) == 1
    assert last is not None
