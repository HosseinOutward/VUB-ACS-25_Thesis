from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import torch

from FL_code.FL_core.codec import Access, BaseProtocol, BaseRoundCodec, CompressionRecord
from FL_code.FL_core.utils import compress_data_list, decompress_data_list


class NSplitCodec(BaseRoundCodec):
    """Round codec that quantizes float values into percentile bins and bin means."""

    record_class = CompressionRecord
    round_name: ClassVar[str] = "NS"
    can_decode_where: ClassVar[Access] = Access.TEMPORAL_TOO

    split_points: int

    def options_to_config(self, points: int) -> None:
        """Set the number of percentile bins used by the split quantizer."""
        self.validate_cfg(points=points)
        self.split_points = points

    @staticmethod
    def validate_cfg(points: int) -> None:
        assert type(points) is int and points > 1, (
            "NS round points option must be an integer greater than 1.")

    def encode(self, delta_vec: torch.Tensor, record: CompressionRecord) -> bytes:
        """Encode a float32 CPU delta into percentile-bin symbols and reconstruction levels."""
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        assert delta_vec.numel() > 0, "NS round cannot encode an empty delta vector."
        assert isinstance(record, CompressionRecord)

        bins, levels = self._bin(delta_vec)
        return compress_data_list({
            "bins": bins,
            "levels": levels,
            "shape": tuple(delta_vec.shape),
        })

    def decode(self, payload: bytes, record: CompressionRecord) -> torch.Tensor:
        """Decode percentile-bin symbols back to their transmitted bin means."""
        payload_dict = decompress_data_list(payload)
        bins = payload_dict["bins"]
        levels = payload_dict["levels"]
        shape = tuple(payload_dict["shape"])
        assert isinstance(bins, torch.Tensor) and isinstance(levels, torch.Tensor)
        return levels[bins.to(torch.int64)].to(torch.float32).reshape(shape)

    def _bin(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        quantiles = torch.arange(1, self.split_points, dtype=x.dtype, device=x.device) / self.split_points
        boundaries = torch.quantile(x, quantiles)
        bins = torch.bucketize(x.contiguous(), boundaries, right=False).to(self._symbol_dtype())

        flat_bins = bins.reshape(-1).to(torch.int64)
        counts = torch.bincount(flat_bins, minlength=self.split_points)
        sums = torch.zeros(self.split_points, dtype=torch.float32)
        sums.scatter_add_(0, flat_bins, x.reshape(-1).to(torch.float32))

        levels = torch.zeros(self.split_points, dtype=torch.float16)
        nonempty = counts > 0
        levels[nonempty] = (sums[nonempty] / counts[nonempty]).to(torch.float16)
        return bins, levels

    def _symbol_dtype(self) -> torch.dtype:
        if self.split_points <= 256:
            return torch.uint8
        if self.split_points <= 32_768:
            return torch.int16
        return torch.int32


class NSplitProtocol(BaseProtocol):
    """Protocol schedule that repeatedly uses the percentile split baseline."""

    warmup_round_codecs: ClassVar[tuple[str, ...]] = ()
    routine_round_codecs: ClassVar[tuple[str, ...]] = ("NS|points=3",)
    protocol_name: ClassVar[str] = "n_split"
    max_per_client_recons_history: ClassVar[int | None] = None

    def create_round_codec(self, round_id: int, client_id: int) -> BaseRoundCodec:
        """Create the configured n-split round codec for this client-round."""
        rc_class, parsed, round_name_full = self._get_curr_round_codec_name(round_id)
        assert rc_class is NSplitCodec and parsed.options is not None
        return NSplitCodec(parsed.options, round_name_full)


if __name__ == "__main__":
    codec = NSplitCodec({"points": 3}, "NS|points=3")
    delta = torch.normal(0, 1, size=(1_000_000,), dtype=torch.float32)
    record = codec.create_r_record(round_id=0, client_id=0)
    reconstruction = codec.decode(codec.encode(delta, record), record)
    print(record.to_dict())
    print(torch.mean((reconstruction - delta).square()).item())
