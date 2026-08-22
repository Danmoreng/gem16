import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_m12_diagnostic_summary_is_compact_and_passes() -> None:
    path = ROOT / "artifacts/m12/diagnostic-summary.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert path.stat().st_size < 32 * 1024
    assert report["milestone"] == "M12"
    assert report["status"] in {
        "diagnostic_pass_acceptance_pending_clean_commit", "acceptance_pass"}
    assert report["artifact_arena_bytes"] == 14_696_668_160
    assert report["fp8_kv_bytes"] == {
        "8192": 188_743_680, "32768": 440_401_920,
        "65536": 775_946_240}
    assert report["traits"]["sliding_layers"] == 25
    assert report["traits"]["full_layers"] == 5
    assert report["traits"]["cross_layer_kv_sharing"] == 0
    assert report["observed_worst"]["relative_l2"] <= 0.07
    assert report["observed_worst"]["cosine"] >= 0.998
    assert report["lifecycle"]["forward_allocation_free"] is True
    assert all(item["status"] == "pass" for item in report["sanitizers"])


def test_m12_acceptance_when_present_is_commit_bound() -> None:
    path = ROOT / "artifacts/m12/acceptance.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["acceptance"] is True
    assert report["status"] == "acceptance_pass"
    assert report["implementation_commit"] == report["code_revision"]
