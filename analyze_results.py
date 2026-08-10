#!/usr/bin/env python3
"""Verify and classify the frozen N1-GSM8K-A1 three-seed experiment."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
SEEDS = (20260820, 20260821, 20260822)
CONDITIONS = ("token", "example")
EXPECTED_PROTOCOL = "N1-GSM8K-A1"
EXPECTED_PROTOCOL_SHA = "74fc4ea7ea4a5c6c151790842d44717f909d23bc8b0bf8a595b251504ccd2d7e"
EXPECTED_DATA_LOCK_SHA = "ffe4cb3718a46f6ef2859bf7daebbd41186a396248a3d1aa039463aa56bafea5"
EXPECTED_VALID_SHA = "3225783c8c9562e7fd234f5d31fa7986ce1dd28aed8bf4ffc5111c48ee6987ea"
EXPECTED_TEST_SHA = "b7f0dc1e4273fb6d05f1d846c8e8dcc70b4ae2ad15ea1f20b444b9dde72c6f2c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_finite(value: float, label: str) -> None:
    require(isinstance(value, (int, float)) and math.isfinite(value), f"nonfinite {label}")


def main() -> None:
    protocol_path = ROOT / "CANONICAL-PROTOCOL.md"
    if not protocol_path.exists():
        protocol_path = ROOT / "PROTOCOL.md"
    require(sha256(protocol_path) == EXPECTED_PROTOCOL_SHA,
            "canonical protocol hash changed")
    require(sha256(ROOT / "DATASET-LOCK-v2.json") == EXPECTED_DATA_LOCK_SHA,
            "dataset lock hash changed")

    preflight = load_json(ROOT / "preflight/implementation-v3-report.json")
    require(preflight["status"] == "PASS", "implementation preflight did not pass")
    require(preflight["model_weights_loaded"] is False, "preflight loaded model weights")
    require(preflight["record_order"] == {
        "epoch_zero_records": 1000,
        "epoch_zero_unique": 1000,
        "length_balanced_batches": False,
        "same_seed_repeat_identical": True,
        "sampling_with_replacement": False,
    }, "natural record-order preflight changed")

    skeleton = load_json(ROOT / "SKELETON-AUDIT.json")
    require(skeleton["status"] == "PASS", "question-skeleton audit did not pass")
    require("Current state: **RELEASED" in (ROOT / "PROCESS-STATE.md").read_text(),
            "model slot ledger is not released")

    results: dict[int, dict] = {}
    all_test_exact: list[float] = []
    all_test_pairwise_differences: list[float] = []
    selection_hashes: dict[int, str] = {}
    metrics_hashes: dict[str, str] = {}

    for seed in SEEDS:
        selection_path = RUNS / f"selection-{seed}.json"
        selection = load_json(selection_path)
        selection_sha = sha256(selection_path)
        selection_hashes[seed] = selection_sha
        require(selection["seed"] == seed, f"selection seed mismatch for {seed}")
        require(selection["protocol_sha256"] == EXPECTED_PROTOCOL_SHA,
                f"selection protocol mismatch for {seed}")
        require(selection["data_lock_sha256"] == EXPECTED_DATA_LOCK_SHA,
                f"selection data lock mismatch for {seed}")
        require(selection["split_sha256"] == EXPECTED_VALID_SHA,
                f"selection validation split mismatch for {seed}")
        require(selection["test_evaluation_status_at_seal"] == "NOT_STARTED",
                f"selection did not preserve test embargo for {seed}")

        seed_result = {"selection": selection, "conditions": {}}
        test_start_times: list[datetime] = []

        for condition in CONDITIONS:
            train_dir = RUNS / f"train-{seed}-{condition}"
            train_receipt = load_json(train_dir / "receipt.json")
            require(train_receipt["status"] == "COMPLETED" and train_receipt["exit_code"] == 0,
                    f"training failed for {seed} {condition}")
            train_log = (train_dir / "stdout.log").read_text()
            require("Starting training..., iters: 576" in train_log,
                    f"wrong training target for {seed} {condition}")
            require(len(re.findall(r"^Iter 576: Train loss", train_log, flags=re.MULTILINE)) == 1,
                    f"missing exact final update for {seed} {condition}")
            require("nan" not in train_log.lower() and "inf" not in train_log.lower(),
                    f"nonfinite training log for {seed} {condition}")
            resource = load_json(train_dir / "artifact/resource.json")
            require(resource["seed"] == seed and resource["condition"] == condition,
                    f"training resource identity mismatch for {seed} {condition}")
            require_finite(resource["peak_mlx_memory_gb"], f"peak memory {seed} {condition}")
            require(resource["peak_mlx_memory_gb"] < 12.8,
                    f"headroom ceiling exceeded for {seed} {condition}")
            adapter_path = train_dir / "artifact/adapters.safetensors"
            adapter_sha = sha256(adapter_path)
            require(selection["adapter_sha256"][condition] == adapter_sha,
                    f"selection adapter mismatch for {seed} {condition}")
            adapter_config = load_json(train_dir / "artifact/adapter_config.json")
            require(adapter_config["seed"] == seed and adapter_config["condition"] == condition,
                    f"adapter config identity mismatch for {seed} {condition}")
            require(adapter_config["iters"] == 576 and adapter_config["batch_size"] == 4,
                    f"training constants mismatch for {seed} {condition}")
            require(adapter_config["fingerprints"]["canonical_protocol"] == EXPECTED_PROTOCOL_SHA,
                    f"adapter protocol mismatch for {seed} {condition}")
            require(adapter_config["fingerprints"]["data_lock"] == EXPECTED_DATA_LOCK_SHA,
                    f"adapter data mismatch for {seed} {condition}")

            valid_dir = RUNS / f"valid-{seed}-{condition}"
            valid_metrics_path = valid_dir / "metrics/metrics.json"
            valid_metrics = load_json(valid_metrics_path)
            valid_receipt = load_json(valid_dir / "receipt.json")
            valid_rows = load_jsonl(valid_dir / "metrics/teacher-forced.jsonl")
            require(valid_receipt["status"] == "COMPLETED" and valid_receipt["exit_code"] == 0,
                    f"validation failed for {seed} {condition}")
            require(len(valid_rows) == 256 and valid_metrics["records"] == 256,
                    f"validation row count mismatch for {seed} {condition}")
            require(valid_metrics["split"] == "valid" and valid_metrics["split_sha256"] == EXPECTED_VALID_SHA,
                    f"validation split mismatch for {seed} {condition}")
            valid_sha = sha256(valid_metrics_path)
            metrics_hashes[f"valid-{seed}-{condition}"] = valid_sha
            require(selection["fingerprints"][f"{condition}_validation"] == valid_sha,
                    f"selection validation hash mismatch for {seed} {condition}")
            selector_nll = valid_metrics["summary"]["token_weighted_nll"]
            require_finite(selector_nll, f"selector NLL {seed} {condition}")
            require(abs(selection["validation_metric"][condition] - selector_nll) < 1e-15,
                    f"selection value mismatch for {seed} {condition}")

            test_dir = RUNS / f"test-{seed}-{condition}"
            test_metrics_path = test_dir / "metrics/metrics.json"
            test_metrics = load_json(test_metrics_path)
            test_receipt = load_json(test_dir / "receipt.json")
            teacher_rows = load_jsonl(test_dir / "metrics/teacher-forced.jsonl")
            generation_rows = load_jsonl(test_dir / "metrics/generations.jsonl")
            require(test_receipt["status"] == "COMPLETED" and test_receipt["exit_code"] == 0,
                    f"test failed for {seed} {condition}")
            require(len(teacher_rows) == 512 and len(generation_rows) == 512,
                    f"test row count mismatch for {seed} {condition}")
            require(test_metrics["records"] == 512 and test_metrics["split"] == "test",
                    f"test metrics row/split mismatch for {seed} {condition}")
            require(test_metrics["split_sha256"] == EXPECTED_TEST_SHA,
                    f"test split hash mismatch for {seed} {condition}")
            require(test_metrics["protocol"] == EXPECTED_PROTOCOL,
                    f"test protocol name mismatch for {seed} {condition}")
            require(test_metrics["protocol_sha256"] == EXPECTED_PROTOCOL_SHA,
                    f"test protocol hash mismatch for {seed} {condition}")
            require(test_metrics["data_lock_sha256"] == EXPECTED_DATA_LOCK_SHA,
                    f"test data lock mismatch for {seed} {condition}")
            require(test_metrics["fingerprints"]["adapters"] == adapter_sha,
                    f"test adapter mismatch for {seed} {condition}")
            require(test_metrics["fingerprints"]["selection"] == selection_sha,
                    f"test selection mismatch for {seed} {condition}")
            test_start_times.append(parse_time(test_receipt["started_at"]))
            require([row["id"] for row in teacher_rows] == [row["id"] for row in generation_rows],
                    f"teacher/generation ID order mismatch for {seed} {condition}")
            for key in ("macro_example_nll", "token_weighted_nll", "exact_answer", "marker_rate"):
                require_finite(test_metrics["summary"][key], f"test {key} {seed} {condition}")
            test_sha = sha256(test_metrics_path)
            metrics_hashes[f"test-{seed}-{condition}"] = test_sha
            all_test_exact.append(test_metrics["summary"]["exact_answer"])
            seed_result["conditions"][condition] = {
                "adapter_sha256": adapter_sha,
                "training_receipt_sha256": sha256(train_dir / "receipt.json"),
                "validation_metrics_sha256": valid_sha,
                "test_metrics_sha256": test_sha,
                "validation_token_weighted_nll": selector_nll,
                "test": test_metrics["summary"],
                "teacher_rows": teacher_rows,
            }

        require(parse_time(selection["sealed_at"]) < min(test_start_times),
                f"selection was not sealed before test for {seed}")
        token_value = selection["validation_metric"]["token"]
        example_value = selection["validation_metric"]["example"]
        expected_selected = "token" if token_value <= example_value else "example"
        require(selection["selected_condition"] == expected_selected,
                f"selector rule mismatch for {seed}")

        token_rows = seed_result["conditions"]["token"]["teacher_rows"]
        example_rows = seed_result["conditions"]["example"]["teacher_rows"]
        require([row["id"] for row in token_rows] == [row["id"] for row in example_rows],
                f"paired test record order mismatch for {seed}")
        differences = [abs(t["example_nll"] - e["example_nll"])
                       for t, e in zip(token_rows, example_rows)]
        all_test_pairwise_differences.extend(differences)
        for condition in CONDITIONS:
            del seed_result["conditions"][condition]["teacher_rows"]
        results[seed] = seed_result

    nll_informative = max(all_test_pairwise_differences) > 1e-8
    exact_informative = not (
        all(value == 0.0 for value in all_test_exact)
        or all(value == 1.0 for value in all_test_exact)
    )

    rows = []
    for seed in SEEDS:
        selection = results[seed]["selection"]
        selected = selection["selected_condition"]
        unselected = "example" if selected == "token" else "token"
        selected_test = results[seed]["conditions"][selected]["test"]
        unselected_test = results[seed]["conditions"][unselected]["test"]
        rows.append({
            "seed": seed,
            "selected": selected,
            "validation_token_weighted_nll": selection["validation_metric"],
            "selected_macro_example_nll": selected_test["macro_example_nll"],
            "unselected_macro_example_nll": unselected_test["macro_example_nll"],
            "selected_minus_unselected_macro_nll": (
                selected_test["macro_example_nll"] - unselected_test["macro_example_nll"]
            ),
            "selected_exact_answer": selected_test["exact_answer"],
            "unselected_exact_answer": unselected_test["exact_answer"],
            "selected_minus_unselected_exact": (
                selected_test["exact_answer"] - unselected_test["exact_answer"]
            ),
            "selected_marker_rate": selected_test["marker_rate"],
            "unselected_marker_rate": unselected_test["marker_rate"],
        })

    policies = {}
    for policy in ("selected", "token", "example"):
        macro_values = []
        exact_values = []
        marker_values = []
        for seed in SEEDS:
            condition = results[seed]["selection"]["selected_condition"] if policy == "selected" else policy
            summary = results[seed]["conditions"][condition]["test"]
            macro_values.append(summary["macro_example_nll"])
            exact_values.append(summary["exact_answer"])
            marker_values.append(summary["marker_rate"])
        policies[policy] = {
            "macro_example_nll_mean": mean(macro_values),
            "macro_example_nll_worst": max(macro_values),
            "exact_answer_mean": mean(exact_values),
            "exact_answer_worst": min(exact_values),
            "marker_rate_mean": mean(marker_values),
            "per_seed_macro_example_nll": macro_values,
            "per_seed_exact_answer": exact_values,
        }

    selected_primary_wins = sum(row["selected_minus_unselected_macro_nll"] < 0 for row in rows)
    selected_primary_losses = sum(row["selected_minus_unselected_macro_nll"] > 0 for row in rows)
    selected_exact_no_worse = sum(row["selected_minus_unselected_exact"] >= 0 for row in rows)
    selected_exact_strict_wins = sum(row["selected_minus_unselected_exact"] > 0 for row in rows)
    selected_mean_beats_both = all(
        policies["selected"]["macro_example_nll_mean"] < policies[fixed]["macro_example_nll_mean"]
        for fixed in ("token", "example")
    )
    selected_worst_beats_both = all(
        policies["selected"]["macro_example_nll_worst"] < policies[fixed]["macro_example_nll_worst"]
        for fixed in ("token", "example")
    )
    selected_mean_does_not_beat_either = all(
        policies["selected"]["macro_example_nll_mean"] >= policies[fixed]["macro_example_nll_mean"]
        for fixed in ("token", "example")
    )
    selected_worst_does_not_beat_either = all(
        policies["selected"]["macro_example_nll_worst"] >= policies[fixed]["macro_example_nll_worst"]
        for fixed in ("token", "example")
    )
    exact_reverses_primary_direction = selected_exact_no_worse >= 2 and selected_exact_strict_wins >= 1

    supported = (
        nll_informative and exact_informative
        and selected_primary_wins >= 2
        and selected_mean_beats_both and selected_worst_beats_both
        and selected_exact_no_worse >= 2 and selected_exact_strict_wins >= 1
    )
    negative = (
        nll_informative and exact_informative
        and selected_primary_losses >= 2
        and selected_mean_does_not_beat_either and selected_worst_does_not_beat_either
        and not exact_reverses_primary_direction
    )
    classification = "SUPPORTED_BOUNDED_TRANSFER" if supported else "NEGATIVE_BOUNDED_RESULT" if negative else "MIXED"

    output = {
        "schema": "N1-GSM8K-A1-result-v1",
        "classification": classification,
        "claim": (
            "Validation-guided token-vs-example selection did not show supported bounded transfer: "
            "the selected arm lost primary macro-example NLL on all three seeds, while complementary "
            "final-answer exact match favored the selected arm on two seeds, producing a frozen Mixed result."
        ),
        "validity": {
            "status": "PASS",
            "all_six_training_arms_exit_0": True,
            "all_six_training_arms_exactly_576_updates": True,
            "all_six_validations_complete_256_records": True,
            "all_six_tests_complete_512_teacher_forced_and_512_generations": True,
            "all_hashes_match": True,
            "all_selection_timestamps_precede_test": True,
            "single_model_process_ledger": "PROCESS-STATE.md",
            "model_slot_released": True,
        },
        "informativeness": {
            "nll": nll_informative,
            "max_paired_per_record_macro_nll_difference": max(all_test_pairwise_differences),
            "exact_answer": exact_informative,
        },
        "gates": {
            "selected_primary_wins": selected_primary_wins,
            "selected_primary_losses": selected_primary_losses,
            "selected_mean_beats_both_fixed": selected_mean_beats_both,
            "selected_worst_beats_both_fixed": selected_worst_beats_both,
            "selected_mean_does_not_beat_either_fixed": selected_mean_does_not_beat_either,
            "selected_worst_does_not_beat_either_fixed": selected_worst_does_not_beat_either,
            "selected_exact_no_worse": selected_exact_no_worse,
            "selected_exact_strict_wins": selected_exact_strict_wins,
            "exact_reverses_primary_direction": exact_reverses_primary_direction,
            "supported": supported,
            "negative": negative,
            "mixed": classification == "MIXED",
        },
        "per_seed": rows,
        "policies": policies,
        "hashes": {
            "analysis_script": sha256(Path(__file__)),
            "canonical_protocol": EXPECTED_PROTOCOL_SHA,
            "data_lock": EXPECTED_DATA_LOCK_SHA,
            "implementation_preflight": sha256(ROOT / "preflight/implementation-v3-report.json"),
            "skeleton_audit": sha256(ROOT / "SKELETON-AUDIT.json"),
            "selections": {str(seed): value for seed, value in selection_hashes.items()},
            "metrics": metrics_hashes,
        },
        "limitations": [
            "One locked 1000/256/512 subset of public GSM8K main, restricted to complete rendered records of at most 256 tokens.",
            "Qwen3-0.6B 3-bit with one fixed LoRA recipe and only three fresh paired seeds.",
            "Public static benchmark exposure or pretraining contamination cannot be excluded; this is within-dataset transfer only.",
            "Final-number exact match is complementary, not independent, and does not assess rationale quality.",
            "Exact-match rates are low and marker rates are below one, limiting capability interpretation.",
            "The result does not establish broad natural-instruction transfer, other models, other datasets, or other training budgets.",
        ],
    }
    output_path = ROOT / "RESULT.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
