"""Regenerate synthetic CSVs and a compact reference from actual OPE computation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import run_example  # noqa: E402


def refresh() -> None:
    output = ROOT / "output" / "reference"
    metrics = run_example(output, seed=41)
    target = ROOT / "examples"
    target.mkdir(exist_ok=True)
    for split in ("train", "validation", "test"):
        (target / f"synthetic_{split}.csv").write_bytes(
            (output / f"synthetic_{split}.csv").read_bytes()
        )
    summary = {
        "schema": "counterledger-example-reference-v1",
        "seed": 41,
        "data_origin": "fresh_generator_v1",
        "target_policy_origin": "deterministic_synthetic_not_llm",
        "clinical_benefit_claimed": False,
        "input_manifest": json.loads((output / "input_manifest.json").read_text()),
        "training_roles": metrics["training_roles"],
        "winner_assessment": metrics["winner_assessment"],
        "simultaneous_family_size": metrics["simultaneous_inference"]["family_size"],
        "comparison": {
            name: {
                "direct": policy["estimates"]["primary"]["hist_gradient_boosting"]["direct"],
                "sdr_cap_10": policy["estimates"]["primary"]["hist_gradient_boosting"][
                    "sdr_cap_10.0"
                ],
                "terminal_raw_ess": policy["support"]["cumulative_raw_ess_by_horizon"][-1],
            }
            for name, policy in metrics["policies"].items()
        },
    }
    (target / "reference.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    refresh()
