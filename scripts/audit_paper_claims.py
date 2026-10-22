"""Audit rounded paper claims against frozen machine-readable evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = Path(os.environ.get("YOLO26_EVIDENCE_ROOT", ROOT)).resolve()
RESULTS_DRAFT = ROOT / "docs/paper/PAPER_RESULTS_DRAFT.md"
METHODS = ROOT / "docs/paper/PAPER_METHODS_AND_CAPTIONS.md"
MANUSCRIPT = ROOT / "docs/paper/THESIS_MANUSCRIPT_DRAFT.md"
TEMPLATE_CHECKLIST = ROOT / "docs/paper/THESIS_TEMPLATE_INTEGRATION_CHECKLIST.md"
SCI_POSITIONING = ROOT / "docs/paper/SCI_SUBMISSION_POSITIONING.md"
REFERENCES = ROOT / "docs/paper/references.bib"
RESEARCH_PLAN = ROOT / "docs/archive/protocols/DISTILLATION_RESEARCH_PLAN.md"
PRE_ROUTER_PROTOCOL = ROOT / "docs/protocols/PRE_INFERENCE_UTILITY_ROUTER_PROTOCOL.md"
SCOUT_ROUTER_PROTOCOL = ROOT / "docs/protocols/SCOUT_DETECTION_VALUE_ROUTER_PROTOCOL.md"
P2P3_RESULTS = ROOT / "results/p2p3-stage20"
RDCI_RESULTS = ROOT / "results/rdci-stage20"
PASCP_RESULTS = ROOT / "results/pascp-stage20"
ROUTER_RESULTS = ROOT / "results/perspective-router-final"
ROUTER_BOOTSTRAP = ROOT / "results/router-paired-bootstrap"
RUNTIME_BENCHMARK = ROOT / "results/integrated-router-runtime/benchmark.json"
SCOUT_ROUTER_SUMMARY = ROOT / "results/scout-router-screen/summary.json"
VISDRONE_ORACLE_GATE = ROOT / "results/visdrone-resolution-oracle/gate.json"


def load_metrics(relative_path: str) -> dict:
    """Load the metrics object from one frozen result JSON."""
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))["metrics"]


def sha256(path: Path) -> str:
    """Return the SHA256 digest of one evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_path(relative_path: str) -> Path:
    """Resolve tracked evidence locally and large ignored evidence from a canonical root."""
    local = ROOT / relative_path
    return local if local.is_file() else EVIDENCE_ROOT / relative_path


def expected_claims() -> dict[Path, list[str]]:
    """Build the rounded claims that must occur in each paper document."""
    baseline = load_metrics("results/baseline/bdd100k-yolo26n-640-size-metrics.json")
    p2 = load_metrics("results/p2/bdd100k-yolo26n-p2-640-size-metrics.json")
    val = load_metrics("results/visdrone-native-e100/yolo26n-e100-size-val.json")
    test = load_metrics("results/visdrone-native-e100/yolo26n-e100-size-test-dev.json")
    summary = json.loads((ROOT / "results/visdrone-multiseed/summary.json").read_text(encoding="utf-8"))
    bootstrap = json.loads((ROOT / "results/bdd-paired-bootstrap/summary.json").read_text(encoding="utf-8"))
    error_analysis = json.loads((ROOT / "results/bdd-error-analysis/manifest.json").read_text(encoding="utf-8"))
    p2p3 = json.loads((P2P3_RESULTS / "summary.json").read_text(encoding="utf-8"))
    rdci = json.loads((RDCI_RESULTS / "summary.json").read_text(encoding="utf-8"))
    pascp = json.loads((PASCP_RESULTS / "summary.json").read_text(encoding="utf-8"))
    router = json.loads((ROUTER_RESULTS / "router-comparison.json").read_text(encoding="utf-8"))
    router_full960 = load_metrics("results/perspective-router-final/b1-full-960.json")
    router_vs_b0 = json.loads((ROUTER_BOOTSTRAP / "learned-vs-b0.json").read_text(encoding="utf-8"))
    router_vs_heuristic = json.loads((ROUTER_BOOTSTRAP / "learned-vs-heuristic.json").read_text(encoding="utf-8"))
    runtime = json.loads(RUNTIME_BENCHMARK.read_text(encoding="utf-8"))
    scout_router = json.loads(SCOUT_ROUTER_SUMMARY.read_text(encoding="utf-8"))
    visdrone_oracle = json.loads(VISDRONE_ORACLE_GATE.read_text(encoding="utf-8"))
    learned_router = router["policies"]["learned_p2_bands"]["metrics"]
    random_router = router["policies"]["hash_random_seed0"]["metrics"]
    heuristic_router = router["policies"]["coarse_small_score_sum"]["metrics"]
    size = summary["aggregate"]["size"]
    p2_small_gain = (p2["small"]["AP50_95"] / baseline["small"]["AP50_95"] - 1) * 100
    test_small_gap = (test["small"]["AP50_95"] / val["small"]["AP50_95"] - 1) * 100
    visdrone_960_small_gain = visdrone_oracle["changes"]["full960_small_ap_relative_percent"]
    oracle_retention = 100 * visdrone_oracle["changes"]["oracle25_small_ap_gain_retention"]
    scout_gain = scout_router["ensemble_relative_utility_capture_percent"]
    shared = [
        f"{baseline['all']['AP50_95']:.4f}",
        f"{p2['small']['AP50_95']:.4f}",
        f"{p2_small_gain:.2f}%",
        f"{size['all']['AP50_95']['mean']:.4f}±{size['all']['AP50_95']['sample_std']:.4f}",
        f"{size['small']['AP50_95']['mean']:.4f}±{size['small']['AP50_95']['sample_std']:.4f}",
        f"{test_small_gap:.2f}%",
    ]
    return {
        RESULTS_DRAFT: shared
        + [
            f"{bootstrap['bootstrap']['all']['delta']['ci95_percentile'][0]:+.5f}",
            f"{bootstrap['bootstrap']['small']['delta']['ci95_percentile'][0]:+.5f}",
            f"{error_analysis['candidate_counts']['p2_repaired_small']['events']:,}",
            f"{error_analysis['candidate_counts']['shared_small_miss']['events']:,}",
            f"{error_analysis['candidate_counts']['p2_added_false_positive']['events']:,}",
            f"{abs(p2p3['groups']['d1']['relative_delta_percent']['small_ap']):.2f}%",
            f"{p2p3['groups']['d2']['relative_delta_percent']['small_ap']:.2f}%",
            f"{p2p3['groups']['d3']['relative_delta_percent']['small_ap']:.2f}%",
            f"{rdci['groups']['R1']['relative_to_R0_percent']['all_ap']:.2f}%",
            f"{rdci['groups']['R1']['relative_to_R0_percent']['small_ap']:.2f}%",
            f"{abs(rdci['groups']['R1']['relative_to_R0_percent']['small_ar']):.2f}%",
            f"{abs(pascp['groups']['R1']['relative_to_R0_percent']['all_ap']):.2f}%",
            f"{abs(pascp['groups']['R1']['relative_to_R0_percent']['small_ap']):.2f}%",
            f"{abs(pascp['groups']['R1']['relative_to_R0_percent']['small_ar']):.2f}%",
            f"{learned_router['all']['AP50_95']:.4f}",
            f"{learned_router['small']['AP50_95']:.4f}",
            f"{learned_router['small']['AR50_95']:.4f}",
            f"{router_vs_b0['bootstrap']['all']['delta']['ci95_percentile'][0]:+.5f}",
            f"{router_vs_b0['bootstrap']['small']['delta']['ci95_percentile'][0]:+.5f}",
            f"{router_vs_heuristic['bootstrap']['all']['delta']['ci95_percentile'][0]:+.5f}",
            f"{router_vs_heuristic['bootstrap']['small']['delta']['ci95_percentile'][1]:+.5f}",
            f"{runtime['summary']['learned_25']['total_seconds']['mean']:.4f}",
            f"{runtime['summary']['full_960']['total_seconds']['mean']:.4f}",
            f"{runtime['summary']['learned_25']['relative_total_percent']['vs_full_960']:.2f}%",
            f"{visdrone_oracle['metrics']['full960']['small_ap']:.6f}",
            f"{visdrone_960_small_gain:.2f}%",
            f"{oracle_retention:.2f}%",
            f"{scout_router['ensemble']['utility_capture']:.5f}",
            f"{scout_router['baselines']['hist_gradient_boosting']['utility_capture']:.5f}",
            f"{scout_gain:.2f}%",
        ],
        METHODS: [
            shared[3],
            shared[4],
            f"{abs(test_small_gap):.2f}%",
            f"{rdci['groups']['R1']['relative_to_R0_percent']['small_ap']:.2f}%",
            f"{abs(rdci['groups']['R1']['relative_to_R0_percent']['small_ar']):.2f}%",
            f"{abs(pascp['groups']['R1']['relative_to_R0_percent']['small_ap']):.2f}%",
            f"{abs(pascp['groups']['R1']['relative_to_R0_percent']['small_ar']):.2f}%",
            "4,061",
            "HistGradientBoosting",
        ],
        MANUSCRIPT: shared[:5]
        + [
            f"{abs(test_small_gap):.2f}%",
            f"{bootstrap['bootstrap']['all']['delta']['ci95_percentile'][0]:+.5f}",
            f"{bootstrap['bootstrap']['small']['delta']['ci95_percentile'][1]:+.5f}",
            f"{error_analysis['candidate_counts']['p2_repaired_small']['events']:,}",
            f"{error_analysis['candidate_counts']['shared_small_miss']['events']:,}",
            f"{error_analysis['candidate_counts']['p2_added_false_positive']['events']:,}",
            f"{learned_router['all']['AP50_95']:.4f}",
            f"{learned_router['small']['AP50_95']:.4f}",
            f"{learned_router['small']['AR50_95']:.4f}",
            f"{router_full960['all']['AP50_95']:.4f}",
            f"{random_router['all']['AP50_95']:.4f}",
            f"{heuristic_router['all']['AP50_95']:.4f}",
            f"{router_vs_b0['bootstrap']['all']['delta']['ci95_percentile'][0]:+.5f}",
            f"{router_vs_b0['bootstrap']['small']['delta']['ci95_percentile'][1]:+.5f}",
            f"{router_vs_heuristic['bootstrap']['all']['delta']['ci95_percentile'][0]:+.5f}",
            f"{router_vs_heuristic['bootstrap']['small']['delta']['ci95_percentile'][1]:+.5f}",
            f"{runtime['summary']['learned_25']['total_seconds']['mean']:.4f}",
            f"{runtime['summary']['learned_25']['relative_total_percent']['vs_full_960']:.2f}%",
            f"{visdrone_960_small_gain:.2f}%",
            f"{oracle_retention:.2f}%",
            f"{scout_router['ensemble']['utility_capture']:.5f}",
            f"{scout_router['baselines']['hist_gradient_boosting']['utility_capture']:.5f}",
            f"{scout_gain:.2f}%",
        ],
    }


def audit(require_external_evidence: bool = False) -> list[str]:
    """Return human-readable failures; an empty list means the audit passed."""
    failures = []
    if not TEMPLATE_CHECKLIST.is_file():
        failures.append("missing thesis template integration checklist")
    if not SCI_POSITIONING.is_file():
        failures.append("missing SCI submission positioning")
    for path, claims in expected_claims().items():
        text = path.read_text(encoding="utf-8")
        failures.extend(f"{path.name}: missing {claim}" for claim in claims if claim not in text)
    for stem in (
        "bdd-size-ap-comparison",
        "visdrone-training-curves",
        "visdrone-multiseed-size-errorbars",
        "visdrone-val-test-size-gap",
        "visdrone-official-class-gap",
    ):
        for suffix in ("png", "pdf"):
            if not (ROOT / "results/paper_figures" / f"{stem}.{suffix}").is_file():
                failures.append(f"missing figure: {stem}.{suffix}")
    for stem in ("p2-repaired-small", "shared-small-miss", "p2-added-false-positive"):
        for suffix in ("png", "pdf"):
            if not (ROOT / "results/bdd-error-analysis" / f"{stem}.{suffix}").is_file():
                failures.append(f"missing BDD error figure: {stem}.{suffix}")
    manifest = json.loads((P2P3_RESULTS / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in manifest["files"].items():
        relative_path = (
            "runs/detect/phase4-distill-stage20/matrix_status.json"
            if name == "matrix_status.json"
            else f"results/p2p3-stage20/{name}"
        )
        path = evidence_path(relative_path)
        if not path.is_file():
            if require_external_evidence:
                failures.append(f"missing P2-to-P3 evidence: {name}")
        elif sha256(path) != expected_hash:
            failures.append(f"P2-to-P3 evidence hash mismatch: {name}")
    rdci_manifest = json.loads((RDCI_RESULTS / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in rdci_manifest["files"].items():
        relative_path = (
            "runs/detect/phase5-rdci-stage20/matrix_status.json"
            if name == "matrix_status.json"
            else f"results/rdci-stage20/{name}"
        )
        path = evidence_path(relative_path)
        if not path.is_file():
            if require_external_evidence:
                failures.append(f"missing RDCI evidence: {name}")
        elif sha256(path) != expected_hash:
            failures.append(f"RDCI evidence hash mismatch: {name}")
    pascp_manifest = json.loads((PASCP_RESULTS / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in pascp_manifest["files"].items():
        relative_path = (
            "runs/detect/phase6-pascp-stage20/matrix_status.json"
            if name == "matrix_status.json"
            else f"results/pascp-stage20/{name}"
        )
        path = evidence_path(relative_path)
        if not path.is_file():
            if require_external_evidence:
                failures.append(f"missing PASCP evidence: {name}")
        elif sha256(path) != expected_hash:
            failures.append(f"PASCP evidence hash mismatch: {name}")
    router_manifest = json.loads((ROUTER_RESULTS / "manifest.json").read_text(encoding="utf-8"))
    for name, expected_hash in router_manifest["files"].items():
        path = evidence_path(name)
        if not path.is_file():
            if require_external_evidence:
                failures.append(f"missing perspective-router evidence: {name}")
        elif sha256(path) != expected_hash:
            failures.append(f"perspective-router evidence hash mismatch: {name}")
    for name, expected_hash in {
        "learned-vs-b0.json": "4d35907496740cc397acd7ce4d12bd5b82bebaa2868fe3a541c3c8c6ef58ea69",
        "learned-vs-heuristic.json": "e0a330f02ee8e915fd48aa61dc0bdd5aa17d1e43368e3bd5594ed93fffef999a",
    }.items():
        path = ROUTER_BOOTSTRAP / name
        if not path.is_file():
            failures.append(f"missing router Bootstrap evidence: {name}")
        elif sha256(path) != expected_hash:
            failures.append(f"router Bootstrap evidence hash mismatch: {name}")
    expected_runtime_hash = "14706f62eacc7373824c685cd4c3b2825e4c2b6d7a8878929dc7b7c761e82754"
    if not RUNTIME_BENCHMARK.is_file():
        failures.append("missing integrated runtime evidence: benchmark.json")
    elif sha256(RUNTIME_BENCHMARK) != expected_runtime_hash:
        failures.append("integrated runtime evidence hash mismatch: benchmark.json")
    expected_scout_router_hash = "e451d741221812ed8ad5aeb2908b07b4ad3739fa4974b0de1063182e142aa80c"
    if not SCOUT_ROUTER_SUMMARY.is_file():
        failures.append("missing ScoutDet-RV internal gate evidence: summary.json")
    elif sha256(SCOUT_ROUTER_SUMMARY) != expected_scout_router_hash:
        failures.append("ScoutDet-RV internal gate evidence hash mismatch: summary.json")
    paper_text = (
        RESULTS_DRAFT.read_text(encoding="utf-8")
        + METHODS.read_text(encoding="utf-8")
        + MANUSCRIPT.read_text(encoding="utf-8")
        + RESEARCH_PLAN.read_text(encoding="utf-8")
        + PRE_ROUTER_PROTOCOL.read_text(encoding="utf-8")
        + SCOUT_ROUTER_PROTOCOL.read_text(encoding="utf-8")
        + SCI_POSITIONING.read_text(encoding="utf-8")
    )
    cited = set(re.findall(r"\[@([A-Za-z0-9_:-]+)\]", paper_text))
    available = set(re.findall(r"@[A-Za-z]+\{([^,]+),", REFERENCES.read_text(encoding="utf-8")))
    failures.extend(f"missing bibliography entry: {key}" for key in sorted(cited - available))
    return failures


def main() -> None:
    """Run the audit as a command-line check."""
    require_external = os.environ.get("YOLO26_REQUIRE_EXTERNAL_EVIDENCE") == "1"
    failures = audit(require_external_evidence=require_external)
    if failures:
        raise SystemExit("\n".join(failures))
    scope = "including external ignored evidence" if require_external else "tracked evidence and available external files"
    print(f"Paper claim audit passed: {scope} verified.")


if __name__ == "__main__":
    main()
