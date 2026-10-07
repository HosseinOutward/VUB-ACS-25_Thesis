"""Regression checks for execution-history-independent CUDA precision and provenance."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

from FL_code.FL_core.explicit_cache import CacheEntry, TorchCodec, content_hash
from FL_code.FL_core.models import initialize_model
from FL_code.FL_core.numerical_policy import NumericalPolicy
from FL_code.FL_core.trace import Trace, _seeded
from FL_code.FL_core.utils import set_global_seed
from FL_code.cancer_protocol.history import HistoryVector
from FL_code.cancer_protocol.prior_code import PriorCalculator
from FL_code.cancer_protocol.round_codecs import Oracle_RoundCodec
from FL_code.cancer_protocol.wz_quantizer import WZQuantizerCancer, batch_loop, new_rnn_model, wz_model_training_loop
from FL_code.run_fl import FLConfig


@pytest.fixture(autouse=True)
def restore_policy() -> Iterator[None]:
    """Leave the test process's backend and RNG state untouched."""
    with NumericalPolicy.current().scope(), torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        yield


@pytest.mark.parametrize("precision", ["highest", "high", "medium"])
@pytest.mark.parametrize("fail", [False, True])
def test_nested_scope_restores_complete_state(precision: str, fail: bool) -> None:
    """Restore the exact matmul mode and every controlled setting, including after exceptions."""
    torch.set_float32_matmul_precision(precision)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    caller = NumericalPolicy.current()
    outer = NumericalPolicy()
    try:
        with outer.scope():
            assert NumericalPolicy.current() == outer
            with NumericalPolicy(float32_matmul_precision="highest", cudnn_allow_tf32=False).scope():
                assert not torch.backends.cuda.matmul.allow_tf32
            assert NumericalPolicy.current() == outer
            if fail:
                raise RuntimeError("precision scope failure")
    except RuntimeError as error:
        assert str(error) == "precision scope failure"
    assert NumericalPolicy.current() == caller
    assert torch.backends.cuda.matmul.allow_tf32 == (precision != "highest")


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_initializer_has_no_precision_side_effect(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct real FL wrappers without mutating precision, on CPU or CUDA."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setattr("FL_code.FL_core.models._create_backbone", lambda *_args: torch.nn.Linear(3, 2))
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = False
    caller = NumericalPolicy.current()
    initialize_model(FLConfig(protocol="identity", num_classes=2), torch.device(device))
    assert NumericalPolicy.current() == caller


def test_inference_restores_policy_and_retains_training_provenance() -> None:
    """Inference uses its own policy without relabeling deliberately imported weights."""
    policy = NumericalPolicy(float32_matmul_precision="highest", cudnn_allow_tf32=False)
    model = new_rnn_model(3, 2, 1, False, 4, 1, numerical_policy=policy)
    original = {"numerical_runtime": {"float32_matmul_precision": "high"}, "artifact": "imported-baseline"}
    model.training_provenance = original
    caller = NumericalPolicy.current()

    def evaluate(start: int, end: int) -> torch.Tensor:
        assert NumericalPolicy.current() == policy
        return torch.ones(end - start)

    assert batch_loop(evaluate, model, 5, 2).tolist() == [1.] * 5
    assert model.training_provenance is original
    assert NumericalPolicy.current() == caller

    def fail(_start: int, _end: int) -> torch.Tensor:
        assert NumericalPolicy.current() == policy
        raise RuntimeError("inference failure")

    with pytest.raises(RuntimeError, match="inference failure"):
        batch_loop(fail, model, 5, 2)
    assert NumericalPolicy.current() == caller


def test_cache_match_uses_policy_and_runtime(tmp_path: Path) -> None:
    """Numerically different fits cannot share a cache identity, even with identical AMP settings."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    enabled = NumericalPolicy().runtime(device, mixed_precision=False)
    disabled = NumericalPolicy(float32_matmul_precision="highest", cudnn_allow_tf32=False).runtime(
        device, mixed_precision=False)
    assert enabled["torch_version"] == str(torch.__version__)
    assert enabled["gpu"] == (torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    assert CacheEntry(tmp_path, enabled).identity != CacheEntry(tmp_path, disabled).identity
    assert content_hash(enabled) != content_hash(enabled | {"torch_version": "another-build"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_trace_cache_hit_miss_and_failure_restore_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace generation and cache loads leave identical caller precision before every yielded round."""
    features = (torch.arange(24).float().reshape(8, 3) / 10).share_memory_()
    labels = (torch.arange(8) % 2).share_memory_()
    monkeypatch.setattr("FL_code.FL_core.trace.precompute_dataset_to_shared", lambda *_args: (features, labels, None))
    monkeypatch.setattr("FL_code.FL_core.models._create_backbone", lambda *_args: torch.nn.Linear(3, 2))
    trace = Trace(FLConfig(protocol="identity", num_classes=2, num_clients=2, num_loader_workers=0,
                          local_epochs=1, batch_size=4, mixed_precision=False),
                  dataset_revision="precision-regression", device=torch.device("cuda:0"))
    entry: CacheEntry[list[tuple[int, tuple[torch.Tensor, ...]]]] = CacheEntry(tmp_path, trace.identity(), TorchCodec())
    checkpoint: CacheEntry[dict[str, Any]] = CacheEntry(tmp_path / "continuation", trace.identity(), TorchCodec())
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.allow_tf32 = False
    caller = NumericalPolicy.current()
    with _seeded(43, trace.device):
        assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
        assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert NumericalPolicy.current() == caller
    cold = entry.resolve(lambda: list(trace.stream(0, 0, checkpoint, 1)))
    assert NumericalPolicy.current() == caller
    assert checkpoint.load()["provenance"] == trace.identity()

    def unexpected_miss() -> list[tuple[int, tuple[torch.Tensor, ...]]]:
        pytest.fail("Expected a gradient cache hit")

    hot = entry.resolve(unexpected_miss)
    assert NumericalPolicy.current() == caller
    for cold_delta, hot_delta in zip(cold[0][1], hot[0][1], strict=True):
        assert torch.equal(cold_delta, hot_delta)
    stream = trace.stream(1, 1, checkpoint)
    next(stream)
    assert NumericalPolicy.current() == caller
    stream.close()
    assert NumericalPolicy.current() == caller

    def fail(*_args: Any) -> torch.Tensor:
        assert torch.backends.cuda.matmul.allow_tf32
        raise RuntimeError("trace training failure")

    monkeypatch.setattr("FL_code.FL_core.trace.train_client_update", fail)
    with pytest.raises(RuntimeError, match="trace training failure"):
        list(trace.stream(0, 0))
    assert NumericalPolicy.current() == caller


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("precision", ["high", "highest"])
@pytest.mark.parametrize("seed", [39, 40])
def test_seeded_o_checkpoints_and_rd_match_after_cache_hit_or_miss(
    precision: str, seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeat real O fitting on one GPU; all 17 epoch weights, symbols and three RD points match."""
    import FL_code.cancer_protocol.wz_quantizer as quantizer_module

    device = torch.device("cuda:0")
    policy = NumericalPolicy(float32_matmul_precision=precision, cudnn_allow_tf32=precision == "high")
    x = torch.sin(torch.arange(128).float() / 9) + 3
    options = dict(bins_per_plane=2, num_planes=3, sd_slices=(slice(0, 128),), hidden_dim=8, layers=1,
                   train_epochs=68, train_sample_size=128, train_batch_size=64,
                   warm_training=True, warm_retrain_repeats=1, quantizer_train_repeats=1,
                   training_log_every_epochs=1, numerical_policy=policy)
    set_global_seed(17)
    source = Oracle_RoundCodec(options, "O")
    source.build_quantizer(None, [HistoryVector(x * 0.9)])
    weights = {key: value.clone() for key, value in source.quantizer.coding_model.state_dict().items()}
    entry: CacheEntry[torch.Tensor] = CacheEntry(tmp_path, {"data": x}, TorchCodec())
    checkpoints: list[str] = []
    original_metrics = WZQuantizerCancer.training_metrics

    def metrics(model: Any, inputs: Any, side_info: Any) -> str:
        assert NumericalPolicy.current() == policy
        checkpoints.append(content_hash(model.state_dict()))
        return original_metrics(model, inputs, side_info)

    monkeypatch.setattr(WZQuantizerCancer, "training_metrics", staticmethod(metrics))
    # Tiny untrained fixture weights need not pass research candidate screening; fitting is real.
    monkeypatch.setattr(PriorCalculator, "candidate_rates", classmethod(lambda *_args: (0., 1.)))
    observed_losses: list[float] = []

    def fit(*args: Any, **kwargs: Any) -> float:
        loss = wz_model_training_loop(*args, **kwargs)
        observed_losses.append(loss)
        return loss

    monkeypatch.setattr(quantizer_module, "wz_model_training_loop", fit)
    runs: list[tuple[list[str], torch.Tensor, torch.Tensor, tuple[float, ...]]] = []
    for ambient in ("highest", "medium"):
        torch.set_float32_matmul_precision(ambient)
        torch.backends.cudnn.allow_tf32 = ambient != "highest"
        caller = NumericalPolicy.current()

        def generate() -> torch.Tensor:
            with _seeded(43, device):
                initialize_model(FLConfig(protocol="identity", num_classes=10), device)
            return x

        target = entry.resolve(generate)
        assert NumericalPolicy.current() == caller
        set_global_seed(17)
        codec = Oracle_RoundCodec(options, "O")
        codec.build_quantizer(None, [HistoryVector(x * 0.9)])
        codec.quantizer.load_warm_start_state(weights)
        set_global_seed(seed)
        checkpoints.clear()
        codec.pre_encode_training_steps(target)
        assert NumericalPolicy.current() == caller
        bins, _, metadata = codec.quantizer.encoding_process(target)
        reconstruction = codec.quantizer.decoding_stages_process((bins, metadata))
        assert NumericalPolicy.current() == caller
        provenance = codec.quantizer.training_provenance
        assert provenance is not None and provenance["float32_matmul_precision"] == precision
        assert provenance["gpu"] == torch.cuda.get_device_name(device)
        runs.append((checkpoints.copy(), bins, reconstruction, codec.quantizer.quantizer_prior_stage_rates))
    assert len(runs[0][0]) == len(runs[1][0]) == 17
    assert runs[0][0] == runs[1][0] and observed_losses[0] == observed_losses[1]
    assert torch.equal(runs[0][1], runs[1][1]) and torch.equal(runs[0][2], runs[1][2])
    assert len(runs[0][3]) == 3 and runs[0][3] == runs[1][3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_training_failure_restores_precision() -> None:
    """A loss exception cannot leak the training policy into later work."""
    from FL_code.cancer_protocol.wz_quantizer import WZcfgQuant

    cfg = WZcfgQuant(bins_per_plane=2, num_planes=3, train_epochs=1, train_sample_size=4)
    model = new_rnn_model(3, 2, 1, False, 4, 1)
    torch.set_float32_matmul_precision("medium")
    caller = NumericalPolicy.current()

    def fail(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        assert NumericalPolicy.current() == cfg.numerical_policy
        raise RuntimeError("loss failure")

    with pytest.raises(RuntimeError, match="loss failure"):
        wz_model_training_loop(fail, model, model.parameters(), (torch.ones(4, 1),), torch.ones(4, 1), cfg)
    assert NumericalPolicy.current() == caller
