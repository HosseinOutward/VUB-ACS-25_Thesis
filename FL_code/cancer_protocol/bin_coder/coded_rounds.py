"""Registered WZ round codecs whose staged indices use a concrete transport."""

from __future__ import annotations

from typing import ClassVar

from FL_code.FL_core.codec import Access

from ..round_codecs import Oracle_RoundCodec, T_RoundCodec
from .oracle_sw_mixin import OracleSWRoundMixin
from .rans_mixin import RansRoundMixin


class TRansRoundCodec(RansRoundMixin, T_RoundCodec):
    """Temporal WZ quantization with temporal transport and both available rANS reports."""

    round_name: ClassVar[str] = "TR"
    transport_prior_access = Access.TEMPORAL_TOO


class OracleRansRoundCodec(RansRoundMixin, Oracle_RoundCodec):
    """Server-SI WZ quantization followed by server-SI rANS."""

    round_name: ClassVar[str] = "OR"
    transport_prior_access = Access.SERVER_ONLY


class OracleSWRoundCodec(OracleSWRoundMixin, Oracle_RoundCodec):
    """Oracle WZ quantization followed by the study-only oracle SW transport."""

    round_name: ClassVar[str] = "OSW"
    transport_prior_access = Access.SERVER_ONLY


__all__ = [
    "OracleRansRoundCodec",
    "OracleSWRoundCodec",
    "TRansRoundCodec",
]
