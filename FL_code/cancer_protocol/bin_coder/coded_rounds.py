"""Registered WZ round codecs whose staged indices use a concrete transport."""

from __future__ import annotations

from typing import ClassVar

from ..round_codecs import Oracle_RoundCodec, T_RoundCodec
from .oracle_sw_mixin import OracleSWRoundMixin
from .rans_mixin import RansRoundMixin


class TRansRoundCodec(RansRoundMixin, T_RoundCodec):
    """Temporal WZ quantization followed by rANS using the granted oracle prior.

    The prior is fitted from the transmitted symbols and supplied to the decoder out of band;
    its parameters are therefore excluded from the staged payload byte count.
    """

    round_name: ClassVar[str] = "TR"
    code_with_oracle_prior = True


class OracleRansRoundCodec(RansRoundMixin, Oracle_RoundCodec):
    """Oracle WZ quantization followed by rANS using the granted oracle prior.

    The prior is fitted from the transmitted symbols and supplied to the decoder out of band;
    its parameters are therefore excluded from the staged payload byte count.
    """

    round_name: ClassVar[str] = "OR"
    code_with_oracle_prior = True


class TRansOraclePriorRoundCodec(RansRoundMixin, T_RoundCodec):
    """rANS using a source-fitted oracle prior and the quantizer's decoder SI.

    The prior is fitted to every payload, including every frozen-round reuse, so this codec is a
    study-only oracle rather than an operational entropy coder.
    """

    round_name: ClassVar[str] = "TRR"
    code_with_oracle_prior = True


class OracleSWRoundCodec(OracleSWRoundMixin, Oracle_RoundCodec):
    """Oracle WZ quantization followed by the study-only oracle SW transport."""

    round_name: ClassVar[str] = "OSW"


class OracleSWOraclePriorRoundCodec(OracleSWRoundMixin, Oracle_RoundCodec):
    """Oracle SW using a source-fitted oracle prior and the quantizer's decoder SI.

    The prior is fitted to every payload, including every frozen-round reuse, in addition to the
    SW transport's oracle rate search.
    """

    round_name: ClassVar[str] = "OSWR"
    code_with_oracle_prior = True


__all__ = [
    "OracleRansRoundCodec",
    "OracleSWOraclePriorRoundCodec",
    "OracleSWRoundCodec",
    "TRansOraclePriorRoundCodec",
    "TRansRoundCodec",
]
