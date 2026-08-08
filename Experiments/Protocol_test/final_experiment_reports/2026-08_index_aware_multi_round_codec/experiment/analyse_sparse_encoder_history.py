from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import logging
from pathlib import Path
import sys

import pandas as pd

REPORT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.Protocol_test.analysis import (
    RECORD_FILE_NAME,
    averaged_curve,
    build_curves,
    plot_comparison,
    plot_variant_gaps,
    validate_columns,
    variant_gap_table,
)
from FL_code.FL_core.rd_metrics import rd_curve_gap


RECORDS_DIR = REPORT_ROOT / "results/sparse_encoder_history/records"
PLOTS_DIR = REPORT_ROOT / "results/sparse_encoder_history/analysis"
MANIFEST_PATH = RECORDS_DIR / "test_config.json"
SPARSE_COLUMNS = {
    "codec_config",
    "training_round_offsets",
    "training_problem_weights",
    "case_id",
    "variant_start",
    "client_repeat",
    "start_round",
}
ENCODER_HISTORY_COMPARISONS = (
    ("OM_FE", "OM"),
    ("TM_FE", "TM"),
    ("TM_FE", "OM_FE"),
)


def canonical_protocol(codec: str) -> str:
    """Return one protocol identity per codec despite its many sparse training starts."""
    return "wz_cancer" if codec == "O" else f"wz_cancer|replace_O_with='{codec}'"


def load_sparse_records(
    csv_paths: list[Path], manifest: Mapping[str, object]
) -> dict[Path, pd.DataFrame]:
    """Load partial aggregates of complete cases and express absolute rounds as frozen age."""
    frozen_rounds = manifest["frozen_rounds"]
    codec_configs = manifest["codec_configs"]
    cases = manifest["cases"]
    assert isinstance(frozen_rounds, int)
    assert isinstance(codec_configs, list) and all(
        isinstance(config, dict) for config in codec_configs
    )
    assert isinstance(cases, list) and all(isinstance(case, dict) for case in cases)
    configured_by_name = {str(config["name"]): config for config in codec_configs}
    records_by_path: dict[Path, pd.DataFrame] = {}
    for csv_path in csv_paths:
        records = pd.read_csv(csv_path)
        validate_columns(records, csv_path)
        missing = SPARSE_COLUMNS - set(records.columns)
        assert not missing, f"{csv_path} is missing sparse metadata columns {sorted(missing)}."
        codecs = records["codec_config"].drop_duplicates().tolist()
        assert codecs == [csv_path.parent.name] and codecs[0] in configured_by_name, (
            f"{csv_path} has codec_config values {codecs}."
        )
        expected_config = configured_by_name[str(codecs[0])]
        for column in ("training_round_offsets", "training_problem_weights"):
            values = records[column].drop_duplicates().tolist()
            assert len(values) == 1 and json.loads(str(values[0])) == expected_config[column], (
                f"{csv_path} does not match manifest field {column}: {values}."
            )
        case_ids = records["case_id"].drop_duplicates().astype(int).tolist()
        assert case_ids == sorted(set(case_ids)), f"{csv_path} has unordered or repeated case IDs."
        assert set(case_ids) <= set(range(len(cases))), f"{csv_path} has invalid case IDs {case_ids}."
        expected_rows = len(case_ids) * (frozen_rounds + 1)
        assert len(records) == expected_rows, (
            f"{csv_path} has {len(records)} rows; expected {expected_rows}."
        )

        for case_id, case in records.groupby("case_id", sort=True):
            expected_case = cases[int(case_id)]
            for column, manifest_field in (
                ("variant_start", "variant_start"),
                ("client_repeat", "repeat"),
                ("start_round", "start_round"),
                ("client_id", "client_id"),
            ):
                assert case[column].drop_duplicates().tolist() == [expected_case[manifest_field]], (
                    f"Case {case_id} in {csv_path} does not match manifest field {manifest_field}."
                )
            assert len(case) == frozen_rounds + 1
            assert case["start_round"].nunique() == 1
            start_round = int(case["start_round"].iloc[0])
            assert case["round_id"].tolist() == list(
                range(start_round, start_round + frozen_rounds + 1)
            )
            trained = case["round_name_full"].str.split("|").str[0].ne("F")
            assert trained.tolist() == [True, *([False] * frozen_rounds)], (
                f"Case {case_id} in {csv_path} is not one train followed by frozen rounds."
            )

        records = records.assign(
            absolute_round_id=records["round_id"],
            round_id=records["round_id"] - records["start_round"],
            protocol_name_full=canonical_protocol(str(codecs[0])),
        )
        assert records["round_id"].between(0, frozen_rounds).all()
        records_by_path[csv_path] = records
        logging.info(
            "Loaded %s: %d rows, %d cases, frozen ages 0–%d.",
            csv_path,
            len(records),
            records["case_id"].nunique(),
            frozen_rounds,
        )
    return records_by_path


def shared_pair_records(
    records_by_path: Mapping[Path, pd.DataFrame],
) -> dict[Path, pd.DataFrame]:
    """Keep the same completed case IDs on both sides of every available O/T pair."""
    by_codec = {path.parent.name: (path, records) for path, records in records_by_path.items()}
    shared: dict[Path, pd.DataFrame] = {}
    for o_codec, (o_path, o_records) in by_codec.items():
        if "O" not in o_codec:
            continue
        t_codec = o_codec.replace("O", "T")
        if t_codec not in by_codec:
            continue
        t_path, t_records = by_codec[t_codec]
        case_ids = set(o_records["case_id"]) & set(t_records["case_id"])
        if not case_ids:
            continue
        shared[o_path] = o_records[o_records["case_id"].isin(case_ids)]
        shared[t_path] = t_records[t_records["case_id"].isin(case_ids)]
    return shared


def write_encoder_history_gaps(records_by_path: Mapping[Path, pd.DataFrame]) -> None:
    """Write full, training, and frozen RD gaps for the encoder-history ablations."""
    by_codec = {path.parent.name: (path, records) for path, records in records_by_path.items()}
    selectors: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "full": lambda records: records,
        "training": lambda records: records[records["round_id"].eq(0)],
        "frozen": lambda records: records[records["round_id"].ne(0)],
    }
    rows: list[dict[str, float | int | str]] = []
    for candidate, reference in ENCODER_HISTORY_COMPARISONS:
        candidate_path, candidate_records = by_codec[candidate]
        reference_path, reference_records = by_codec[reference]
        for subset, select in selectors.items():
            selected_candidate, selected_reference = (
                select(records) for records in (candidate_records, reference_records)
            )
            rows.append({
                "subset": subset,
                "candidate": candidate,
                "reference": reference,
                "rows": len(selected_candidate),
                **rd_curve_gap(
                    *averaged_curve(selected_candidate, candidate_path),
                    *averaged_curve(selected_reference, reference_path),
                    name=f"{candidate} vs {reference} ({subset})",
                ),
            })
    pd.DataFrame(rows).to_csv(PLOTS_DIR / "encoder_history_gaps.csv", index=False)


def write_repeat_gaps(records_by_path: Mapping[Path, pd.DataFrame]) -> None:
    """Measure the duplicate-repeat spread used as the experiment-local noise check."""
    rows: list[dict[str, float | int | str]] = []
    selectors: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
        "full": lambda records: records,
        "training": lambda records: records[records["round_id"].eq(0)],
        "frozen": lambda records: records[records["round_id"].ne(0)],
    }
    for path, records in records_by_path.items():
        for subset, select in selectors.items():
            repeat_1, repeat_2 = (
                select(records[records["client_repeat"].eq(repeat)])
                for repeat in (1, 2)
            )
            rows.append({
                "subset": subset,
                "codec": path.parent.name,
                "rows_per_repeat": len(repeat_1),
                **rd_curve_gap(
                    *averaged_curve(repeat_2, path),
                    *averaged_curve(repeat_1, path),
                    name=f"{path.parent.name} repeat 2 vs repeat 1 ({subset})",
                ),
            })
    pd.DataFrame(rows).to_csv(PLOTS_DIR / "repeat_gaps.csv", index=False)


def main() -> None:
    """Plot sparse multi-start RD curves and paired gaps over quantizer frozen age."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    assert MANIFEST_PATH.is_file(), f"Missing sparse test manifest: {MANIFEST_PATH}."
    manifest = json.loads(MANIFEST_PATH.read_text())
    csv_paths = sorted(RECORDS_DIR.glob(f"*/{RECORD_FILE_NAME}"))
    assert csv_paths, f"No aggregate {RECORD_FILE_NAME} files found under {RECORDS_DIR}."
    expected_codecs = {config["name"] for config in manifest["codec_configs"]}
    available_codecs = {path.parent.name for path in csv_paths}
    logging.info(
        "Analyzing codecs with completed cases %s; still pending %s.",
        sorted(available_codecs),
        sorted(expected_codecs - available_codecs),
    )
    records_by_path = load_sparse_records(csv_paths, manifest)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    write_encoder_history_gaps(records_by_path)
    write_repeat_gaps(records_by_path)

    is_training: Callable[[pd.DataFrame, Path], pd.Series] = (
        lambda df, _path: df["round_name_full"].str.split("|").str[0].ne("F")
    )
    selectors: dict[str, tuple[str, Callable[[pd.DataFrame, Path], pd.DataFrame]]] = {
        "all": (
            "Sparse test: average over all training and frozen attempts",
            lambda df, _path: df,
        ),
        "training": (
            "Sparse test: quantizer-training attempts (frozen age 0)",
            lambda df, path: df[is_training(df, path)],
        ),
        "frozen": (
            "Sparse test: twelve frozen attempts after each training",
            lambda df, path: df[~is_training(df, path)],
        ),
    }
    curves_by_subset = {
        subset: build_curves(records_by_path, subset, selector)
        for subset, (_title, selector) in selectors.items()
    }
    client_curves_by_subset = {
        subset: build_curves(records_by_path, subset, selector, per_client=True)
        for subset, (_title, selector) in selectors.items()
    }
    plot_comparison(
        curves_by_subset,
        client_curves_by_subset,
        {subset: options[0] for subset, options in selectors.items()},
        PLOTS_DIR / "rate_distortion_comparison.png",
    )

    comparable_records = shared_pair_records(records_by_path)
    if comparable_records:
        gaps = variant_gap_table(comparable_records)
        gaps.rename(columns={"round_id": "frozen_age"}).to_csv(
            PLOTS_DIR / "variant_gap_over_frozen_age.csv", index=False
        )
        plot_variant_gaps(
            gaps,
            PLOTS_DIR / "variant_gap_over_frozen_age.png",
            x_label="Frozen age (0 = quantizer-training attempt)",
            title=(
                "Sparse test: T-side rate cost against its paired O side over frozen age\n"
                "positive = T side needs more rate at matched distortion"
            ),
        )
    else:
        logging.warning("No O/T pair has a shared completed case yet; skipping the gap outputs.")
        for stale_output in (
            PLOTS_DIR / "variant_gap_over_frozen_age.csv",
            PLOTS_DIR / "variant_gap_over_frozen_age.png",
        ):
            stale_output.unlink(missing_ok=True)
    logging.info("Completed current sparse-analysis plots in %s.", PLOTS_DIR)


if __name__ == "__main__":
    main()
