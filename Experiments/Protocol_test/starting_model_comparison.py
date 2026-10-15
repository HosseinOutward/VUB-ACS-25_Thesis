"""Matched multi-seed O/T/raw-S measurements from the retained F starting model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import io
import json
from pathlib import Path
import tarfile
from time import perf_counter
from typing import ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from Experiments.Protocol_test.live_plot import watch
from Experiments.Protocol_test.rd_metrics import wmape_to_db
from FL_code.FL_core.codec import Access, record_reconstruction_metrics
from FL_code.FL_core.explicit_cache import PickleCodec, TorchCodec, atomic_output, content_hash
from FL_code.FL_core.utils import payload_size_mib, set_global_seed
from FL_code.cancer_protocol.bin_coder.coded_rounds import OracleRansRoundCodec, TRansRoundCodec
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.round_codecs import CancerRecord, ServerSIPriorTrainingPlan
from FL_code.cancer_protocol.wz_quantizer import DedupedWZQuantizerCancer, PreprocessMetadata, SideInfoMatrix, WZcfgQuant, _SideInfoRows, select_si_rows


class SampleFitConfig(BaseModel):
    """Named raw-S sample and optimizer settings supplied by the local grid plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(pattern=r"^[a-zA-Z0-9_]+$")
    train_epochs: int = Field(ge=1, strict=True)
    lr: float = Field(gt=0)
    tau: float = Field(gt=0)
    sample_size: int = Field(default=3_000_000, ge=1, strict=True)
    train_batch_size: int = Field(default=6_250, ge=1, strict=True)


@dataclass(frozen=True)
class ComparisonFit:
    """One independently seeded fit, with an optional sample-only optimizer override."""

    method: Literal["O", "O_tau025", "T", "S_tau025", "S"]
    seed: int
    sample_fit: SampleFitConfig | None

    @property
    def label(self) -> str:
        """Give each measured seed/configuration a distinct artifact directory."""
        return f"{self.method}{'_' + self.sample_fit.name if self.sample_fit else ''}_seed{self.seed}"


class ComparisonPlan(BaseModel):
    """Named static-gradient fits consumed by the worker and detached supervisor."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    output_root: Path
    reference_root: Path
    fit_device: str = "cuda:0"
    methods: tuple[Literal["O", "O_tau025", "T", "S_tau025", "S"], ...] = Field(min_length=1)
    seeds: tuple[int, ...] = (43, 44, 45)
    sample_size: int = Field(default=3_000_000, ge=1)
    sample_fits: tuple[SampleFitConfig, ...] = ()

    @property
    def fits(self) -> tuple[ComparisonFit, ...]:
        """Expand named sample settings only for S, leaving ordinary controls unchanged."""
        return tuple(ComparisonFit(method, seed, sample_fit)
                     for seed in self.seeds for method in self.methods
                     for sample_fit in ((self.sample_fits or (None,)) if method in {"S", "S_tau025"} else (None,)))

    @property
    def cases(self) -> tuple[str, ...]:
        """Expose scheduled fit labels for supervisor coverage checks."""
        return tuple(fit.label for fit in self.fits)


def load_plan(path: Path) -> ComparisonPlan:
    """Read the fresh comparison plan and resolve its artifact directories."""
    plan = ComparisonPlan.model_validate_json(path.read_text())
    assert plan.seeds and len(set(plan.cases)) == len(plan.cases)
    return plan.model_copy(update={key: (path.parent / getattr(plan, key)).resolve()
                                   for key in ("output_root", "reference_root")})


class StagePriorRecord(CancerRecord):
    """Ordinary codec records with directly measured fine-tuned prior stage rates."""

    temporal_si_prior_stage_rates: list[float] | None = None
    server_si_prior_stage_rates: list[float] | None = None


class StagePriorMixin:
    """Measure each fine-tuned prior on the same symbols, without fitting it again."""

    record_class: ClassVar[type[StagePriorRecord]] = StagePriorRecord

    def fit_additional_priors(self, bins: torch.Tensor, soft_codes: torch.Tensor,
                              metadata: PreprocessMetadata, record: StagePriorRecord) -> None:
        super().fit_additional_priors(bins, soft_codes, metadata, record)
        for access, evaluation in self.prior_evaluations(metadata).items():
            prior = PriorCalculator.compute_prior_from_network(evaluation.model, bins, evaluation.side_information)
            rates = PriorCalculator.compute_code_lengths(prior, bins, self.c_cfg.num_planes).mean(dim=1).tolist()
            field = "temporal_si_prior_stage_rates" if access is Access.TEMPORAL_TOO else "server_si_prior_stage_rates"
            record.update_record_fields(**{field: rates})


class _MeasuredOracleCodec(StagePriorMixin, OracleRansRoundCodec):
    """O transport with extra stage measurements for this static study."""


class _MeasuredTemporalCodec(StagePriorMixin, TRansRoundCodec):
    """Standard five-column T with temporal transport and a full-SI comparison prior."""


class TunedQuantizer(DedupedWZQuantizerCancer):
    """Shared tau override for matched full-target O and raw-sample S experiments."""

    def training_config(self) -> WZcfgQuant:
        """Use the report's recommended tau while retaining O's warm optimizer schedule."""
        return super().training_config().model_copy(update={"tau": 0.25})


class _MeasuredTunedOracleCodec(_MeasuredOracleCodec):
    """Full-target O with exactly the quantizer optimizer configuration used by tuned S."""

    quantizer_class: ClassVar[type[TunedQuantizer]] = TunedQuantizer


class RawSampleQuantizer(TunedQuantizer):
    """Historical raw-S sampling, with full-vector quantization at evaluation."""

    training_sample_size: int
    sample_indices: torch.Tensor | None = None
    sample_fit_config: SampleFitConfig | None = None

    @property
    def train_batch_size(self) -> int:
        """Apply the sample-fit batch override where the training loop reads its batch size."""
        return self.sample_fit_config.train_batch_size if self.sample_fit_config is not None else self.c_cfg.train_batch_size

    def training_config(self) -> WZcfgQuant:
        """Override only quantizer fitting; the native prior still uses the ordinary WZ config."""
        config = super().training_config()
        return (config.model_copy(update=self.sample_fit_config.model_dump(exclude={"name", "sample_size"}))
                if self.sample_fit_config is not None else config)

    def _fit_prepared(self, x_prep: torch.Tensor, side_info: SideInfoMatrix | _SideInfoRows | torch.Tensor) -> None:
        assert x_prep.shape[0] == self.vector_size
        count = min(self.training_sample_size, x_prep.shape[0])
        self.sample_indices = (torch.arange(count) if count == x_prep.shape[0] else torch.from_numpy(np.sort(
            np.random.default_rng(int(torch.randint(0, 2**31, ()).item())).choice(x_prep.shape[0], count, replace=False))))
        # Preserve the historical draw budget when a fixed pool is smaller than an epoch.
        rows = self.sample_indices.repeat((self.c_cfg.train_sample_size + count - 1) // count)
        super()._fit_prepared(x_prep[rows.to(x_prep.device)], select_si_rows(side_info, rows))


class _MeasuredRawSampleCodec(_MeasuredOracleCodec):
    """Free raw training labels, no acquisition stream and no exact decoder replacement."""

    quantizer_class: ClassVar[type[RawSampleQuantizer]] = RawSampleQuantizer
    quantizer: RawSampleQuantizer


def projected_state(codec: _MeasuredOracleCodec | _MeasuredTemporalCodec,
                    baseline: dict[str, torch.Tensor], temporal_columns: tuple[int, ...] | None) -> dict[str, torch.Tensor]:
    """Keep deployed heads and project only T's two SI-dependent input matrices."""
    state = {key: baseline[key] for key in codec.quantizer.coding_model.state_dict()}
    if temporal_columns is not None:
        bins = codec.c_cfg.bins_per_plane
        decoder = state["decoder.input_layers.0.weight"]
        prior = state["conditionalRNN.input_layers.0.weight"]
        state["decoder.input_layers.0.weight"] = torch.cat((decoder[:, :bins], decoder[:, [bins + i for i in temporal_columns]]), dim=1)
        state["conditionalRNN.input_layers.0.weight"] = torch.cat((prior[:, temporal_columns], prior[:, -bins:]), dim=1)
    return state


def run(plan: ComparisonPlan) -> None:
    """Fit the requested seeds from saved artifacts, retaining all measurements and model weights."""
    reference = json.loads((plan.reference_root / "test_config.json").read_text())
    problem_paths = tuple((plan.reference_root / "problem").glob("*.pkl"))
    baseline_paths = tuple((plan.reference_root / "baseline").glob("*.pt"))
    assert len(problem_paths) == len(baseline_paths) == 1, f"Ambiguous reference artifacts: {plan.reference_root}"
    problem = PickleCodec().load(problem_paths[0])
    fitted = TorchCodec().load(baseline_paths[0])
    assert content_hash(problem["target"]) == reference["target_hash"], "Saved target hash changed."
    assert content_hash(tuple(content_hash(value.materialize()) for value in problem["si"])) == reference["si_hash"], "Saved SI hash changed."
    torch.cuda.set_device(torch.device(plan.fit_device))
    config = WZcfgQuant.model_validate(reference["plan"]["quantizer"]).model_copy(update={"num_planes": 3, "warm_training": True})
    root = Path(__file__).resolve().parents[2]
    sources = {str(path.relative_to(root)): path.read_bytes()
               for directory in (root / "FL_code", root / "Experiments/Protocol_test")
               for path in sorted(directory.rglob("*.py"))}
    depth = 5
    client = reference["plan"]["target_client"]
    temporal_columns = tuple(range(client * depth, (client + 1) * depth))
    match = dict(plan=plan.model_dump(mode="json"), reference_manifest=reference,
                 problem_artifact=str(problem_paths[0]), baseline_artifact=str(baseline_paths[0]),
                 baseline_state_hash=content_hash(fitted["state"]), sources=content_hash(sources),
                 temporal_columns=temporal_columns, temporal_initialization="SI-input-column projection; other weights unchanged",
                 full_si_reference_initialization="starting joint prior, 30-epoch warm fit on fixed T symbols",
                 raw_s_semantics="free raw labels; full normalization; full-vector quantization; no exact replacement",
                 numerical_runtime=config.numerical_policy.runtime(torch.device(plan.fit_device), mixed_precision=config.mixed_precision))
    identity = content_hash(match)
    plan.output_root.mkdir(parents=True, exist_ok=True)
    manifest = plan.output_root / "test_config.json"
    if manifest.exists():
        assert json.loads(manifest.read_text())["identity"] == identity, f"Identity mismatch: {manifest}"
    else:
        with atomic_output(manifest) as temporary:
            temporary.write_text(json.dumps(dict(identity=identity, **match), indent=2))
        with tarfile.open(plan.output_root / "sources.tar.gz", "w:gz") as archive:
            for path, data in sources.items():
                info = tarfile.TarInfo(path)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    print(f"IMPORTED N={problem['target'].numel()} SI={len(problem['si'])} baseline={match['baseline_state_hash']}", flush=True)
    for fit in plan.fits:
        seed, method, label = fit.seed, fit.method, fit.label
        directory = plan.output_root / "cases" / label / "target"
        completion = directory / "complete.json"
        if completion.exists():
            assert json.loads(completion.read_text())["identity"] == identity, f"Identity mismatch: {completion}"
            continue
        options = config.model_dump(exclude={"norm_slices"}) | {"sd_slices": problem["slices"], "rans_lanes": 2048}
        codec = (_MeasuredTemporalCodec(options, "TR") if method == "T" else
                 _MeasuredRawSampleCodec(options, "OR") if method in {"S", "S_tau025"} else
                 _MeasuredTunedOracleCodec(options, "OR") if method == "O_tau025" else _MeasuredOracleCodec(options, "OR"))
        si = [problem["si"][i] for i in temporal_columns] if method == "T" else list(problem["si"])
        codec.build_quantizer(None, si)
        state = projected_state(codec, fitted["state"], temporal_columns if method == "T" else None)
        codec.quantizer.load_warm_start_state(state)
        if method == "T":
            codec._server_si_prior_side_information = list(problem["si"])
            codec._server_si_prior_training_plan = ServerSIPriorTrainingPlan(
                initial_states=({key: fitted["state"][key] for key in codec.quantizer.coding_model.copy_for_prior_retraining().state_dict()},),
                parent_slots=(), attempts_per_parent=config.prior_train_repeats)
        if method in {"S", "S_tau025"}:
            codec.quantizer.training_sample_size = fit.sample_fit.sample_size if fit.sample_fit else plan.sample_size
            codec.quantizer.sample_fit_config = fit.sample_fit
        record = codec.create_r_record(reference["plan"]["target_round"], client)
        record.model_size = problem["target"].numel()
        set_global_seed(seed)
        started = perf_counter()
        codec.pre_encode_training_steps(problem["target"])
        torch.cuda.synchronize()
        record.pre_encode_training_seconds = perf_counter() - started
        started = perf_counter()
        payload = codec.encode(problem["target"], record)
        torch.cuda.synchronize()
        record.encode_seconds = perf_counter() - started
        started = perf_counter()
        reconstruction = codec.decode(payload, record)
        torch.cuda.synchronize()
        record.decode_seconds = perf_counter() - started
        record.total_seconds = record.pre_encode_training_seconds + record.encode_seconds + record.decode_seconds
        record.final_bytes = payload_size_mib(payload)
        record.update_record_fields(**record_reconstruction_metrics(problem["target"], reconstruction))
        assert record.symbol_error_rate == 0.0, f"Symbol errors in {label}"
        directory.mkdir(parents=True, exist_ok=True)
        weights = dict(joint=codec.joint_quantizer_state, native=codec.quantizer.coding_model.state_dict())
        if method == "T":
            weights["full_si_prior"] = codec._server_si_prior_model.state_dict()
        if method in {"S", "S_tau025"}:
            weights["sample_indices"] = codec.quantizer.sample_indices
        with atomic_output(directory / "weights.pt") as temporary:
            torch.save(weights, temporary)
        (directory / "compression_records.csv").unlink(missing_ok=True)
        record.append_record_to_csv(directory)
        result = dict(identity=identity, method=method, fit_seed=seed,
                      sample_fit=fit.sample_fit.model_dump() if fit.sample_fit is not None else None, baseline_seed=reference["plan"]["seed"],
                      initial_state_hash=content_hash(state), quantizer_si_columns=len(si),
                      training_config=codec.quantizer.training_config().model_dump(mode="json", exclude={"norm_slices"}), record=record.to_dict())
        with atomic_output(completion) as temporary:
            temporary.write_text(json.dumps(result, indent=2))
        print(f"COMPLETED {label}: WMAPE={record.wmape:.6f}% native_rANS={record.coder_rate:.9f} fullSI_rANS={record.server_si_rans_rate:.9f} seconds={record.total_seconds:.1f}", flush=True)
        codec.release_round_resources()
        del codec, reconstruction, payload, record, weights
        gc.collect()
        torch.cuda.empty_cache()

def render(output_root: Path, destination: Path) -> None:
    """Draw all completed seed curves in disposable live-plot processes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots()
    for path in sorted((output_root / "cases").glob("*/*/complete.json")):
        result = json.loads(path.read_text())
        record = result["record"]
        axes.plot(np.cumsum(record["server_si_rans_stage_rates"]), wmape_to_db(np.array(record["stage_wmapes"])), "o-",
                  label=path.parent.parent.name)
    axes.set(xlabel="Full-SI rANS (bits/float)", ylabel="WMAPE (dB)", title="Static target: completed multi-seed fits")
    if axes.lines:
        axes.legend()
    figure.tight_layout()
    figure.savefig(destination / "seed_rd.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Run the experiment or provide the supervisor's live plotting interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--runner-pid", type=int)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.runner_pid is not None:
        assert args.control_dir is not None
        watch(args.path, args.runner_pid, args.control_dir, plot_module=__spec__.name, source_paths=(Path(__file__),))
    elif args.out is not None:
        render(args.path, args.out)
    else:
        run(load_plan(args.path))


if __name__ == "__main__":
    main()
