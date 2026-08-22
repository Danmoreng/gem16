import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_m13_diagnostic_summary_is_compact_and_proceeds() -> None:
    path = ROOT / "artifacts/m13/diagnostic-summary.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert path.stat().st_size < 64 * 1024
    assert report["milestone"] == "M13"
    assert report["early_quality_decision"] == "proceed"
    assert all(report["gates"].values())
    assert report["memory"]["weight_arena_bytes"] == 14_696_668_160
    assert report["memory"]["kv_cache_bytes"] == 440_401_920


def test_m13_acceptance_when_present_is_clean_commit_bound() -> None:
    path = ROOT / "artifacts/m13/acceptance.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["acceptance"] is True
    assert report["status"] == "acceptance_pass"
    assert report["implementation_commit"] == report["code_revision"]
