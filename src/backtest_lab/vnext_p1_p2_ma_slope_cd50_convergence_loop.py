from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CORE_OUT = ROOT / "outputs/vnext_p1_p2_layer4_primary80_individual_MA_slope_CD50_action_legs_20260715"
LOOP_OUT = ROOT / "outputs/vnext_p1_p2_MA_slope_CD50_bounded_closure_convergence_20260715"
RADAR_OUTPUTS = Path(r"C:\Users\zergv\Documents\Codex\2026-05-23\ai-stock-rotation-radar-https-docs\outputs")
CORE_MODULE = "backtest_lab.vnext_p1_p2_primary80_ma_slope_cd50_action_legs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_manifest() -> None:
    files = sorted(path for path in LOOP_OUT.iterdir() if path.is_file() and path.name != "manifest.json")
    write_json(
        LOOP_OUT / "manifest.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": [{"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
            "future_data_violation_count": 0,
        },
    )


def stop(state: dict, reason: str) -> None:
    state.update({"status": "stopped", "stop_reason": reason, "updated_at": datetime.now(timezone.utc).isoformat()})
    write_json(LOOP_OUT / "checkpoint.json", state)
    (LOOP_OUT / "current_step.txt").write_text(f"stopped:{reason}\n", encoding="utf-8")
    write_manifest()


def run_core() -> None:
    subprocess.run([sys.executable, "-X", "utf8", "-m", CORE_MODULE], cwd=ROOT, check=True)


def run_radar(adapter: Path, iteration: int, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(adapter),
            "--iteration",
            str(iteration),
            "--core-output",
            str(CORE_OUT),
            "--output",
            str(output),
        ],
        check=True,
    )


def validate_closure(output: Path, frontier_rows: int, frontier_hash: str, atomic_rows: int) -> None:
    readiness = json.loads((output / "readiness_for_core_rechain.json").read_text(encoding="utf-8"))
    if readiness.get("frontier_authority_rows") != frontier_rows:
        raise RuntimeError("Radar closure frontier row count differs from Core authority")
    if readiness.get("frontier_closed_rows") != frontier_rows or readiness.get("frontier_exact_blocked_rows") != 0:
        raise RuntimeError("Radar closure did not close the exact frontier")
    if readiness.get("incumbent_network_download_rows") != 0:
        raise RuntimeError("incumbent local-only rows were used as network authority")
    if readiness.get("provisional_gap_rows_used_as_download_authority") != 0:
        raise RuntimeError("provisional gaps were used as network authority")
    if readiness.get("atomic_policy_blocker_rows") != atomic_rows:
        raise RuntimeError("atomic policy authority changed during Radar closure")
    guard = pd.read_csv(output / "authority_scope_guard_audit.csv")
    frontier_guard = guard.loc[guard.scope.eq("frontier_exact_legs")]
    if len(frontier_guard) != 1 or int(frontier_guard.iloc[0].rows) != frontier_rows:
        raise RuntimeError("Radar authority guard does not match Core frontier")
    write_json(output / "core_frontier_authority_hash.json", {"sha256": frontier_hash, "rows": frontier_rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar-adapter", type=Path, required=True)
    parser.add_argument("--start-iteration", type=int, default=9)
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()
    LOOP_OUT.mkdir(parents=True, exist_ok=True)
    state = {"status": "running", "completed_iterations": [], "frontier_hashes": []}
    checkpoint = LOOP_OUT / "checkpoint.json"
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("status") == "running":
            state = prior
    write_manifest()

    previous_ready = state.get("last_ready_legs")
    previous_classified = state.get("last_classified_rows")
    previous_atomic = state.get("last_atomic_policy_blockers")
    for offset in range(args.max_iterations):
        iteration = args.start_iteration + offset
        (LOOP_OUT / "current_step.txt").write_text(f"core_rechain_iteration_{iteration:03d}\n", encoding="utf-8")
        run_core()
        readiness = json.loads((CORE_OUT / "readiness_for_action_leg_first.json").read_text(encoding="utf-8"))
        frontier_path = CORE_OUT / "p1_p2_MA_slope_CD50_frontier_official_raw_gap_ledger.csv"
        frontier = pd.read_csv(frontier_path)
        frontier_hash = sha256(frontier_path)
        ready_legs = int(readiness["execution_leg_rows"])
        classified = int(readiness["incumbent_no_observation_rows"] - readiness["incumbent_analysis_unclassified_rows"])
        if len(frontier) == 0:
            stop(state, "frontier_zero")
            return
        if frontier_hash in state["frontier_hashes"]:
            stop(state, "frontier_hash_repeated")
            return
        if previous_ready == ready_legs and previous_classified == classified:
            stop(state, "ready_legs_and_classification_no_progress")
            return
        atomic_rows = int(readiness["atomic_policy_blockers"])
        if previous_atomic is not None and atomic_rows > previous_atomic:
            stop(state, "new_atomic_policy_blocker")
            return
        state["frontier_hashes"].append(frontier_hash)
        state["current_iteration"] = iteration
        write_json(checkpoint, state)
        write_manifest()
        radar_out = RADAR_OUTPUTS / f"radar_vnext_p1_p2_ma_slope_cd50_action_leg_frontier_iteration_{iteration:03d}_20260715"
        (LOOP_OUT / "current_step.txt").write_text(f"radar_bounded_closure_iteration_{iteration:03d}\n", encoding="utf-8")
        run_radar(args.radar_adapter, iteration, radar_out)
        validate_closure(radar_out, len(frontier), frontier_hash, atomic_rows)
        state["completed_iterations"].append(iteration)
        previous_ready, previous_classified, previous_atomic = ready_legs, classified, atomic_rows
        state.update({
            "last_ready_legs": ready_legs,
            "last_classified_rows": classified,
            "last_atomic_policy_blockers": atomic_rows,
        })
        write_json(checkpoint, state)
        write_manifest()
    stop(state, "safe_iteration_limit_reached")


if __name__ == "__main__":
    main()
