import json
from pathlib import Path

import pandas as pd

from contracts import portable_file_sha256
from synthetic import generate_data


def test_published_inputs_match_generator_and_hashes():
    root = Path(__file__).resolve().parents[1]
    reference = json.loads((root / "examples/reference.json").read_text())
    assert reference["clinical_benefit_claimed"] is False
    data = generate_data(seed=reference["seed"])
    for split, generated in data.items():
        path = root / "examples" / f"synthetic_{split}.csv"
        pd.testing.assert_frame_equal(pd.read_csv(path), generated, check_exact=False, rtol=1e-12)
        assert (
            portable_file_sha256(path)
            == (reference["input_manifest"]["splits"][split]["canonical_csv_sha256"])
        )


def test_text_inventory_has_no_binary_assets():
    from scripts.check_release import check

    assert len(check()) >= 25


def test_csv_hash_contract_normalizes_real_newlines(tmp_path):
    paths = []
    for index, payload in enumerate((b"a,b\n1,2\n", b"a,b\r\n1,2\r\n", b"a,b\r1,2\r")):
        path = tmp_path / f"case-{index}.csv"
        path.write_bytes(payload)
        paths.append(path)
    assert len({portable_file_sha256(path) for path in paths}) == 1
