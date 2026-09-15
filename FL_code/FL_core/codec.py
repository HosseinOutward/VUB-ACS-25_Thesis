from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
import csv
import tempfile
import time
from typing import TYPE_CHECKING, Any, ClassVar

import torch
import numpy as np

from FL_code.FL_core.utils import (
    ParsedConfigurableName,
    payload_size_mib,
    compress_data_list,
    decompress_data_list,
    parse_configurable_name,
)

if TYPE_CHECKING:
    from .utils import StateDictManager


DEFAULT_PROTOCOL_CACHE_DIR = Path("_/data/protocol_cache")


def _synchronized_time() -> float:
    """Return the current timer value after pending CUDA work completes."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class Access(Enum):
    """Where a round codec's reconstruction may be decoded, and so who may retain it."""
    NONE = "none"
    SERVER_ONLY = "server_only"
    TEMPORAL_TOO = "temporal_too"


# --- Record Row Data --- #
class CompressionRecord:
    """Base metrics row for one client-round, produced by a concrete round codec."""

    round_id: int
    client_id: int

    round_name_full: str
    protocol_name_full: str | None = None

    model_size: int | None = None
    setup_seconds: float | None = None
    pre_encode_training_seconds: float | None = None
    encode_seconds: float | None = None
    decode_seconds: float | None = None
    total_seconds: float | None = None
    # MiB, despite the name, like encoder_size/decoder_size/meta_data_size. The name is kept
    # because it is a column in every recorded run; `payload_size_mib` is what produces it.
    final_bytes: float | None = None
    global_eval_metrics: dict[str, float] | None = None
    worker_eval_metrics: dict[str, dict[str, float]] | None = None

    mse: float | None = None
    mape: float | None = None
    mspe_sqrt: float | None = None
    w_mean_of_vec: float | None = None
    wmape: float | None = None
    wmspe_sqrt: float | None = None

    def __init__(
        self,
        round_id: int,
        client_id: int,
        round_name_full: str,
    ) -> None:
        self.round_id: int = round_id
        self.client_id: int = client_id
        self.round_name_full: str = round_name_full

    def to_dict(self) -> dict[str, Any]:
        """Convert record attributes to a flat CSV-ready dictionary."""
        skipped_fields = {
            "global_eval_metrics", "worker_eval_metrics"}
        result: dict[str, Any] = {
            key: value
            for key, value in vars(self).items()
            if not key.startswith("_") and key not in skipped_fields
        }
        # Add global eval metrics with prefix
        if self.global_eval_metrics is not None:
            for key, value in self.global_eval_metrics.items():
                result[f'global_eval_{key}'] = value

        # Add worker eval metrics with split prefix (e.g., train_loss, test_acc)
        if self.worker_eval_metrics is not None:
            for split, metrics in self.worker_eval_metrics.items():
                for metric_key, metric_value in metrics.items():
                    result[f'{split}_{metric_key}'] = metric_value

        return result

    def update_record_fields(self, **updates: Any) -> None:
        """Update unset record fields with known non-null values."""
        unknown_fields = {field_name for field_name in updates if not hasattr(self, field_name)}
        assert not unknown_fields, f"Unknown record fields: {tuple(unknown_fields)}."
        for field_name, value in updates.items():
            assert getattr(self, field_name) is None, (
                f"CompressionRecord.{field_name} is already set before update.")
            assert value is not None, f"CompressionRecord.{field_name} replacement must not be None."
            setattr(self, field_name, value)

    def save_artifacts(self, save_dir: Path) -> None:
        """Write non-CSV artifacts attached by a concrete record; no-op by default."""

    def append_record_to_csv(self, save_dir: Path) -> None:
        """Append record to CSV file, expanding the header when new metrics appear."""
        save_dir.mkdir(exist_ok=True, parents=True)

        csv_file = save_dir / "compression_records.csv"
        record_dict = self.to_dict()
        fieldnames = list(record_dict)
        rows_to_rewrite = None

        if csv_file.exists():
            with csv_file.open(newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or ())
                new_fieldnames = [key for key in record_dict if key not in fieldnames]
                if new_fieldnames:
                    fieldnames += new_fieldnames
                    rows_to_rewrite = [*reader, record_dict]

        if rows_to_rewrite is not None:
            with tempfile.NamedTemporaryFile(
                'w', newline='', dir=save_dir, prefix=f".{csv_file.stem}.",
                suffix=".tmp", delete=False
            ) as temp_file:
                temp_path = Path(temp_file.name)
                writer = csv.DictWriter(temp_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_to_rewrite)
            temp_path.replace(csv_file)
            return

        with csv_file.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(record_dict)


# --- Round Codecs and Protocols --- #
class BaseRoundCodec:
    """One-round compressor selected by a protocol schedule."""

    record_class: ClassVar[type[CompressionRecord]]
    round_name: ClassVar[str]
    can_decode_where: ClassVar[Access]
    round_name_full: str

    def __init_subclass__(cls) -> None:
        """Require concrete round codecs to declare their record type and round acronym.

        Underscore-prefixed subclasses are abstract intermediates and skip validation.
        can_decode_where may instead be declared as an instance annotation when the
        access level is only known per instance.
        """
        super().__init_subclass__()
        if cls.__name__.startswith("_"):
            return
        assert issubclass(cls.record_class, CompressionRecord)
        assert cls.round_name
        assert (isinstance(getattr(cls, "can_decode_where", None), Access)
                or "can_decode_where" in getattr(cls, "__annotations__", {}))

    def __init__(
        self,
        cfg_options: Mapping[str, Any] | None,
        round_name_full: str,
        **protocol_inputs: Any,
    ) -> None:
        self.round_name_full = round_name_full
        if cfg_options:
            self.validate_cfg(**cfg_options)
            self.options_to_config(**cfg_options)

    def create_r_record(self, round_id: int, client_id: int) -> CompressionRecord:
        """Create this round codec's metrics record for one client-round compression."""
        assert self.record_class is not CompressionRecord, (
            "Declare a CompressionRecord subclass as record_class; the base record is not usable directly.")
        return self.record_class(
            round_id=round_id,
            client_id=client_id,
            round_name_full=self.round_name_full,
        )

    def options_to_config(self, **options: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def validate_cfg(**options: Any) -> None:
        raise NotImplementedError

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> Any:
        raise NotImplementedError

    def pre_encode_training_steps(self, delta_vec: torch.Tensor) -> None:
        """Perform training that requires the current raw vector before encoding it."""

    def decode(self, payload: Any, record: CompressionRecord) -> torch.Tensor:
        raise NotImplementedError


class IdentityRoundCodec(BaseRoundCodec):
    """Round codec that reconstructs the original delta exactly."""

    class IdentityRecord(CompressionRecord):
        """Compression record for one identity client-round."""

    record_class = IdentityRecord
    round_name = "I"
    can_decode_where = Access.TEMPORAL_TOO

    def options_to_config(self, **options: Any) -> None:
        """Reject identity-round options."""
        assert not options, f"I round does not accept options: {tuple(options)}."

    @staticmethod
    def validate_cfg(**options: Any) -> None:
        assert not options, f"I round does not accept options: {tuple(options)}."

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> torch.Tensor:
        """Return the original float32 CPU delta as the identity payload."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        return delta_vec

    def decode(self, payload: torch.Tensor, record: CompressionRecord) -> torch.Tensor:
        """Return the identity payload as the reconstruction."""
        return payload


class BaseProtocol:
    """Protocol schedule that selects the round codec used for each training round."""

    warmup_round_codecs: ClassVar[tuple[str, ...]]
    routine_round_codecs: ClassVar[tuple[str, ...]]
    protocol_name: ClassVar[str]
    protocol_name_full: str
    _round_codec_classes: dict[str, type[BaseRoundCodec]]
    sd_slices: Sequence[slice] | None
    cache_dir: Path | None
    cache_overwrite: bool
    _schedule_placed: bool = False

    def __init_subclass__(cls) -> None:
        """Require concrete protocols to declare their warmup and routine round codec plans.

        Underscore-prefixed subclasses are abstract intermediates and skip validation.
        """
        super().__init_subclass__()
        if cls.__name__.startswith("_"):
            return
        assert isinstance(cls.warmup_round_codecs, tuple)
        assert isinstance(cls.routine_round_codecs, tuple)
        assert cls.routine_round_codecs
        assert cls.protocol_name

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        protocol_name_full: str | None = None,
        sd_slices: Sequence[slice] | None = None,
        cache_dir: Path | None = None,
        cache_overwrite: bool = False,
    ) -> None:
        self.protocol_name_full = protocol_name_full or type(self).protocol_name
        if options:
            self.options_to_config(**options)
        self._round_codec_classes = self._validate_round_codecs()
        self.sd_slices = sd_slices
        self.cache_dir = cache_dir
        self.cache_overwrite = cache_overwrite

    def options_to_config(self, **options: Any) -> Any:
        raise NotImplementedError

    def after_decode(
        self, reconst: torch.Tensor, record: CompressionRecord, round_codec: BaseRoundCodec
    ) -> None:
        """Hand one decoded reconstruction back to the protocol; stateless schedules ignore it."""

    def set_schedule(self, start_round: int, round_codecs: tuple[str, ...]) -> None:
        """Place one training round at `start_round`, then reuse it as frozen forever after.

        Called once on a fresh protocol. The training token is the last warmup entry; routine is
        the single `"F"` because cycling the caller's whole tuple would periodically retrain.
        """
        assert not self._schedule_placed, (
            "set_schedule may be installed only once on a fresh protocol.")
        assert isinstance(start_round, int) and start_round >= 0
        assert isinstance(round_codecs, tuple) and all(
            isinstance(round_codec, str) and round_codec for round_codec in round_codecs
        )
        assert len(round_codecs) >= 2, (
            "round_codecs must supply a training token followed by at least one 'F' round.")
        assert round_codecs[0] != "F", "The first schedule token must train a codec, not 'F'."
        assert round_codecs[1:] == ("F",) * (len(round_codecs) - 1), (
            f"Every round after the training token must be the literal frozen marker 'F'; "
            f"got {round_codecs[1:]!r}."
        )
        self.warmup_round_codecs = ("I",) * start_round + (round_codecs[0],)
        self.routine_round_codecs = ("F",)
        self._round_codec_classes = self._validate_round_codecs()
        self._schedule_placed = True

    def _get_curr_round_codec_name(
        self, round_id: int
    ) -> tuple[type[BaseRoundCodec], ParsedConfigurableName, str]:
        if round_id < len(self.warmup_round_codecs):
            round_name_full = self.warmup_round_codecs[round_id]
        else:
            routine_idx = (round_id - len(self.warmup_round_codecs)) % len(self.routine_round_codecs)
            round_name_full = self.routine_round_codecs[routine_idx]
        parsed = parse_configurable_name(round_name_full, "round codec")

        rc_class = self._round_codec_classes[round_name_full]
        return rc_class, parsed, round_name_full

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        raise NotImplementedError

    def config_snapshot(self) -> dict[str, Any]:
        """Return the protocol schedule and explicit options used by a records run."""
        parsed_protocol = parse_configurable_name(self.protocol_name_full, "protocol")
        return {
            "implementation": f"{type(self).__module__}.{type(self).__qualname__}",
            "protocol_name_full": self.protocol_name_full,
            "options": parsed_protocol.options,
            "warmup_round_codecs": self.warmup_round_codecs,
            "routine_round_codecs": self.routine_round_codecs,
            "round_codecs": {
                round_name: {
                    "implementation": (
                        f"{round_codec.__module__}.{round_codec.__qualname__}"
                    ),
                    "options": parse_configurable_name(round_name, "round codec").options,
                }
                for round_name, round_codec in sorted(self._round_codec_classes.items())
            },
        }

    def _validate_round_codecs(self) -> dict[str, type[BaseRoundCodec]]:
        round_codec_classes: dict[str, type[BaseRoundCodec]] = {}
        for round_name_full in set(self.warmup_round_codecs + self.routine_round_codecs):
            parsed = parse_configurable_name(round_name_full, "round codec")
            round_codec_class = _single_named_subclass(BaseRoundCodec, "round_name", parsed.name)
            if parsed.options:
                round_codec_class.validate_cfg(**parsed.options)
            round_codec_classes[round_name_full] = round_codec_class
        return round_codec_classes


def create_protocol(
    protocol_name: str,
    sd_slices: Sequence[slice] | None = None,
    cache_dir: Path | None = None,
    cache_overwrite: bool = False,
) -> BaseProtocol:
    """Create a validated protocol schedule."""
    from FL_code.cancer_protocol import cancer_protocol
    from FL_code.cancer_protocol.bin_coder import coded_rounds as _coded_rounds
    from FL_code.other_protocols import n_split_protocol
    parsed = parse_configurable_name(protocol_name, "protocol")
    return _single_named_subclass(BaseProtocol, "protocol_name", parsed.name)(
        parsed.options,
        protocol_name,
        sd_slices=sd_slices,
        cache_dir=cache_dir,
        cache_overwrite=cache_overwrite,
    )


def _single_named_subclass(base_class: type, name_attr: str, name: str) -> type:
    # A set, not a list: a class mixing two descendants of `base_class` is reachable by more than
    # one inheritance path, and counting paths would reject legitimate mixin composition.
    def all_subclasses(parent: type) -> set[type]:
        return {
            subclass
            for direct_subclass in parent.__subclasses__()
            for subclass in (direct_subclass, *all_subclasses(direct_subclass))
        }

    # Underscore-prefixed intermediates are abstract and inherit their parent's name; the same
    # convention __init_subclass__ uses to skip them has to hold when resolving a name.
    matches = [
        subclass for subclass in all_subclasses(base_class)
        if getattr(subclass, name_attr, None) == name and not subclass.__name__.startswith("_")]
    assert len(matches) == 1, (
        f"Expected one {base_class.__name__} subclass with {name_attr}={name!r}; got {len(matches)}.")
    return matches[0]


def parse_and_validate_protocol(protocol_name: str) -> str:
    """Validate a configured protocol name and return it unchanged."""
    create_protocol(protocol_name)
    return protocol_name


class IdentityProtocol(BaseProtocol):
    """Lossless protocol used to generate reusable uncompressed federated-learning traces."""

    warmup_round_codecs: ClassVar[tuple[str, ...]] = ()
    routine_round_codecs: ClassVar[tuple[str, ...]] = ("I",)
    protocol_name: ClassVar[str] = "identity"

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        """Create the identity codec selected for the given client-round."""
        round_codec_class, parsed, round_name_full = self._get_curr_round_codec_name(round_id)
        return round_codec_class(parsed.options, round_name_full)


def record_reconstruction_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, float]:
    """Return distortion metrics for one reconstructed delta."""
    assert original.shape == reconstructed.shape, (
        f"Reconstruction shape mismatch: got {tuple(reconstructed.shape)}, expected {tuple(original.shape)}.")
    error = reconstructed - original
    mse = torch.mean(error ** 2).item()
    w_mean_of_vec = torch.abs(original).mean().item()
    assert w_mean_of_vec > 0, "Weight Gradients are suspiciously all zero."
    metrics: dict[str, float] = {
        "mse": mse,
        "mape": torch.mean(torch.abs(error) / (torch.abs(original) + 1e-8)).item() * 100,
        "mspe_sqrt": torch.sqrt(torch.mean(error ** 2 / (original ** 2 + 1e-8))).item() * 100,
        "w_mean_of_vec": w_mean_of_vec,
        "wmape": torch.mean(torch.abs(error)).item() / w_mean_of_vec * 100,
        "wmspe_sqrt": float(np.sqrt(mse)) / w_mean_of_vec * 100,
    }
    return metrics


def simulate_compression(
    protocol: BaseProtocol, delta_vec: torch.Tensor, client_id: int, round_id: int,
    sd_manager: StateDictManager, save_dir: Path,
    server_eval_metrics: dict[str, float], worker_eval_metrics: Sequence[float],
    metric_keys: Sequence[str]
) -> torch.Tensor:
    """Simulate client encoding and server decoding for one flattened delta."""
    model_size = sd_manager.param_count
    num_metrics = len(metric_keys)
    expected_len = num_metrics * 2
    assert len(worker_eval_metrics) == expected_len

    total_start = _synchronized_time()
    start_time = _synchronized_time()
    round_codec = protocol.create_round_codec(round_id, client_id)
    setup_seconds = _synchronized_time() - start_time

    record = round_codec.create_r_record(round_id, client_id)

    start_time = _synchronized_time()
    round_codec.pre_encode_training_steps(delta_vec)
    pre_encode_training_seconds = _synchronized_time() - start_time

    start_time = _synchronized_time()
    payload = round_codec.encode(delta_vec, record)
    encode_seconds = _synchronized_time() - start_time

    start_time = _synchronized_time()
    reconstructed = round_codec.decode(payload, record)
    decode_seconds = _synchronized_time() - start_time
    assert reconstructed.numel() == model_size
    protocol.after_decode(reconstructed.detach().cpu(), record, round_codec)

    final_bytes = payload_size_mib(payload)
    recons_error_and_metric = record_reconstruction_metrics(delta_vec, reconstructed)
    total_seconds = _synchronized_time() - total_start
    record.update_record_fields(
        protocol_name_full=protocol.protocol_name_full,
        model_size=model_size,
        global_eval_metrics=server_eval_metrics,
        worker_eval_metrics={
            'train': {key: worker_eval_metrics[i] for i, key in enumerate(metric_keys)},
            'test': {key: worker_eval_metrics[i + num_metrics] for i, key in enumerate(metric_keys)},
        },
        final_bytes=final_bytes,
        setup_seconds=setup_seconds,
        pre_encode_training_seconds=pre_encode_training_seconds,
        encode_seconds=encode_seconds,
        decode_seconds=decode_seconds,
        total_seconds=total_seconds,
        **recons_error_and_metric,
    )

    record.append_record_to_csv(save_dir)
    record.save_artifacts(save_dir)

    return reconstructed


# --- Demo Round Codec and Protocol --- #
class RCDemo(BaseRoundCodec):
    """Round codec that serializes raw float32 deltas with the shared gzip transport."""

    round_cfg: bool

    class RecordDemo(CompressionRecord):
        comp_report: int | None = None
        random_field: int | None = None
    record_class = RecordDemo
    round_name = "DemoRC"
    can_decode_where = Access.TEMPORAL_TOO

    def create_r_record(self, round_id: int, client_id: int) -> RCDemo.RecordDemo:
        """Create a demo metrics record with a random demo-only field."""
        raw_record = super().create_r_record(round_id, client_id)
        assert isinstance(raw_record, RCDemo.RecordDemo)
        raw_record.random_field = int(np.random.randint(0, 100))
        return raw_record

    def options_to_config(self, compReport: bool) -> None:
        """Set whether demo records include the compression-report metric."""
        self.round_cfg = compReport

    @staticmethod
    def validate_cfg(compReport: bool) -> None:
        assert isinstance(compReport, bool), "DemoRC compReport option must be a boolean."

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> bytes:
        """Encode one raw float32 delta with pickle then gzip."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        payload = compress_data_list(delta_vec)
        if self.round_cfg:
            record.update_record_fields(comp_report=1)
        return payload

    def decode(self, payload: bytes, record: CompressionRecord) -> torch.Tensor:
        """Decode one raw float32 delta from pickle plus gzip."""
        reconstructed = decompress_data_list(payload)
        return reconstructed


class DemoProtocol(BaseProtocol):
    """Protocol schedule that repeatedly uses the raw demo round codec."""
    warmup_round_codecs: ClassVar[tuple[str, ...]] = ()
    routine_round_codecs: ClassVar[tuple[str, ...]] = ("DemoRC|compReport",)
    protocol_name: ClassVar[str] = "demo"

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        rc_class, parsed, round_name_full = self._get_curr_round_codec_name(round_id)
        return rc_class(parsed.options, round_name_full)
