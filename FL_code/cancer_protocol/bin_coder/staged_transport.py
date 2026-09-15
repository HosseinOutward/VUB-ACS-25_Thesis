"""Shared round-codec plumbing for transports that replace staged WZ indices."""

from __future__ import annotations

from math import isclose
from typing import Any, ClassVar

import torch

from FL_code.FL_core.utils import decompress_data_list

from ..brent_wz_models import EncoderDecoderLayeredRNN
from ..round_codecs import CancerRecord
from ..wz_quantizer import PreprocessMetadata, new_rnn_model


class StagedIndexTransportMixin:
    """Encode, account for, and decode the staged indices emitted by a WZ round codec."""

    # This model is fitted on the symbols being sent, so codecs enabling it are explicitly
    # named oracle codecs rather than ordinary transport alternatives.
    code_with_oracle_prior: ClassVar[bool] = False
    _oracle_prior: EncoderDecoderLayeredRNN
    # The out-of-band oracle grant: decoder-side parameters of the source-fitted prior, handed to
    # a decoder explicitly (never serialized into the wire payload and never inferred from this
    # encoder object's own `_oracle_prior` instance -- see `decode`).
    oracle_prior_grant: dict[str, torch.Tensor] | None = None

    def encode_staged_indices(self, bins: torch.Tensor, metadata: PreprocessMetadata) -> Any:
        """Return the transport payload for a staged symbol tensor."""
        raise NotImplementedError

    def decode_staged_indices(self, payload: Any, metadata: PreprocessMetadata) -> torch.Tensor:
        """Recover staged symbols using decoder-available state."""
        raise NotImplementedError

    def payload_bits(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, payload: Any
    ) -> tuple[list[int], int]:
        """Return the coded bits attributed to each stage and fixed payload overhead bits."""
        raise NotImplementedError

    def symbol_error_rate(
        self, bins: torch.Tensor, metadata: PreprocessMetadata, payload: Any
    ) -> float:
        """Return the encoder-known symbol error rate for this transport payload."""
        raise NotImplementedError

    def coder_prior_model(self) -> Any:
        """Return the prior model evaluated identically by transport encoder and decoder."""
        return self._oracle_prior if self.code_with_oracle_prior else self.quantizer.coding_model

    def conditional_code_lengths(self, *args: Any, **kwargs: Any) -> tuple[Any, EncoderDecoderLayeredRNN]:
        """Fit the source-fitted prior and record its explicit out-of-band decoder grant."""
        code_lengths, self._oracle_prior = super().conditional_code_lengths(*args, **kwargs)
        # The grant, not `self._oracle_prior` itself: a separate decoder object must recover this
        # from an explicit hand-off, not from this encoder's retained model instance.
        self.oracle_prior_grant = self._oracle_prior.decoder_state_dict()
        return code_lengths, self._oracle_prior

    def add_prior_to_record(self, *args: Any, **kwargs: Any) -> None:
        """Fit an oracle coder prior against the quantizer's exact decoder side information."""
        if not self.code_with_oracle_prior:
            return
        # A protocol may estimate diagnostics with wider server SI. An actual coder must evaluate
        # the same model with the same SI shape at both ends, including temporal TRR rounds.
        diagnostic_side_information = self._prior_side_information
        self._prior_side_information = None
        try:
            super().add_prior_to_record(*args, **kwargs)
        finally:
            self._prior_side_information = diagnostic_side_information

    def _load_oracle_prior(
        self, metadata: PreprocessMetadata, state_dict: dict[str, torch.Tensor]
    ) -> EncoderDecoderLayeredRNN:
        """Rebuild the granted source-fitted prior from its out-of-band decoder-side parameters.

        Reconstructs the same architecture `conditional_code_lengths` trained (see
        `PriorCalculator.make_trained_prior_model`), rather than reusing `self._oracle_prior` set
        at encode time: a genuinely separate decoder object has no such attribute and must recover
        it from the explicit `oracle_prior_grant` hand-off instead.
        """
        side_info_size = self.quantizer.side_info_tensor(metadata).shape[1]
        model = new_rnn_model(
            num_planes=self.c_cfg.num_planes,
            bins_per_plane=self.c_cfg.bins_per_plane,
            side_info_size=side_info_size,
            marginal=False,
            hidden_dim=self.c_cfg.hidden_dim,
            layers=self.c_cfg.layers,
            shared_heads=self.c_cfg.shared_heads,
        )
        # The transmitted state is decoder-side only (see EncoderDecoderLayeredRNN.decoder_state_dict);
        # the freshly built encoder half is never evaluated by get_priors.
        model.load_state_dict(state_dict, strict=False)
        return model.eval()

    def get_encod_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        """Replace plain staged indices with a transport payload and record wire metrics."""
        outer_payload = super().get_encod_payload(delta_vec, record)
        bins, metadata = outer_payload["payload_content"]
        payload = self.encode_staged_indices(bins, metadata)
        stage_bits, overhead_bits = self.payload_bits(bins, metadata, payload)
        assert len(stage_bits) == bins.shape[0]
        positions = bins.shape[1]
        # Fixed per-payload overhead (header, lane flush, or transmitted rate plan) is charged
        # entirely to stage 0 rather than spread across stages, so coder_stage_rates always sums
        # exactly to coder_rate and coder_overhead_rate names an already-included component of it,
        # not an extra charge.
        coder_rate = (sum(stage_bits) + overhead_bits) / positions
        coder_stage_rates = [
            (bits + (overhead_bits if stage == 0 else 0)) / positions
            for stage, bits in enumerate(stage_bits)
        ]
        assert isclose(sum(coder_stage_rates), coder_rate, rel_tol=1e-9), (
            f"{self.round_name_full}: coder_stage_rates must sum to coder_rate exactly.")
        record.update_record_fields(
            coder_rate=coder_rate,
            coder_stage_rates=coder_stage_rates,
            coder_overhead_rate=overhead_bits / positions,
            symbol_error_rate=self.symbol_error_rate(bins, metadata, payload),
        )
        outer_payload["payload_content"] = (payload, metadata)
        # The oracle prior is granted out-of-band (see `oracle_prior_grant`): it must not be
        # serialized here, or TRR/OSWR's final_bytes would silently include a whole prior model's
        # weights and stop being comparable to TR/OSW's.
        return outer_payload

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        """Recover staged indices and return the quantizer's final reconstruction."""
        decoded_payload = decompress_data_list(payload)
        transmitted, metadata = decoded_payload["payload_content"]
        if self.code_with_oracle_prior:
            assert self.oracle_prior_grant is not None, (
                f"{self.round_name_full}: decode needs an explicit oracle_prior_grant; "
                "it is never read from the payload or from an encoder's retained _oracle_prior."
            )
            self._oracle_prior = self._load_oracle_prior(metadata, self.oracle_prior_grant)
        bins = self.decode_staged_indices(transmitted, metadata)
        stage_reconstructions = self.quantizer.decoding_stages_process((bins, metadata))
        self.add_stage_distortion_to_record(stage_reconstructions, record)
        return stage_reconstructions[-1]

    @classmethod
    def codec_provenance(cls, **options: Any) -> dict[str, Any]:
        """Layer coder-prior facts and rate-field availability onto the base quantizer facts."""
        facts = super().codec_provenance(**options)
        facts["oracle_prior_granted"] = cls.code_with_oracle_prior
        facts["coder_prior_si"] = (
            "decoder_legal_exact_oracle_grant" if cls.code_with_oracle_prior else "quantizer_decoder_si"
        )
        facts["prior_stage_rates_si"] = (
            "decoder_legal_exact_oracle_grant" if cls.code_with_oracle_prior else None
        )
        facts["oracle_prior_grant_channel"] = (
            "out_of_band_not_in_final_bytes" if cls.code_with_oracle_prior else None
        )
        facts["rate_fields_available"] = facts["rate_fields_available"] | {
            "prior_stage_rates": cls.code_with_oracle_prior,
            "marginal_stage_rates": cls.code_with_oracle_prior,
            "training_prior_stage_rates": cls.code_with_oracle_prior,
            "coder_rate": True,
            "coder_stage_rates": True,
            "coder_overhead_rate": True,
            "symbol_error_rate": True,
        }
        return facts


__all__ = ["StagedIndexTransportMixin"]
