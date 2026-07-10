from __future__ import annotations

from encodings.cp932 import codec
import gc
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import torch
from pydantic import BaseModel, ConfigDict

from FL_code.FL_core.codec import (
    Access,
    BaseProtocol,
    BaseRoundCodec,
    CompressionRecord,
    HistoryEntry,
    ReconstructionHistory,
)
from FL_code.FL_core.utils import ParsedConfigurableName, compress_data_list, decompress_data_list

from .wz_quantizer import DedupedDecodingWZQuantizerCancer, WZQuantizerCancer, WZcfgQuant
from .prior_code import PriorCalculator


def _compressed_size(obj: Any) -> float:
    """Return compressed payload size in MB."""
    return len(compress_data_list(obj)) / (1024 ** 2)


class NewCancerConfig(BaseModel):
    """Configuration shared by NewCancer round codecs and produced by NewCancerProtocol."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    bins_per_plane: int = 8
    num_planes: int = 3
    norm_slices: list[slice] | None = None
    marginal_loss: bool = True
    max_side_info_count: int = 5

class CancerRecord(CompressionRecord):
    """Compression record for one NewCancer client-round."""

    phase: str
    bins_per_plane: int
    num_planes: int
    prior_rate: float
    marginal_rate: float
    encoder_size: float
    decoder_size: float
    meta_data_size: float


# --- Demo Round Codec and Protocol --- #
class _WZRoundCodec(BaseRoundCodec):
    record_class = CancerRecord

    round_name: ClassVar[str] # needs to be set in subclasses
    can_decode_where: ClassVar[Access] # needs to be set in subclasses

    quantizer: WZQuantizerCancer
    marginal_loss_function: bool

    c_cfg: WZcfgQuant
    quantizer_class: type[WZQuantizerCancer] = DedupedDecodingWZQuantizerCancer

    where_quantizer_trained: Literal["server", "worker"] = "server"

    def create_r_record(self, round_id: int, client_id: int) -> CancerRecord:
        raw_record = super().create_r_record(round_id, client_id)
        assert isinstance(raw_record, CancerRecord)
        raw_record.bins_per_plane = self.c_cfg.bins_per_plane
        raw_record.num_planes = self.c_cfg.num_planes
        return raw_record

    def options_to_config(self, bins_per_plane: int, num_planes: int) -> None:
        self.c_cfg = WZcfgQuant(bins_per_plane=bins_per_plane, num_planes=num_planes)

    @staticmethod
    def validate_cfg(bins_per_plane: int, num_planes: int) -> None:
        assert isinstance(bins_per_plane, int) and bins_per_plane > 0
        assert isinstance(num_planes, int) and num_planes > 0

    def get_training_data(self, given_history: ReconstructionHistory, client_id: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        history:dict[int, tuple[HistoryEntry, ...]] = given_history.view(Access.SERVER_ONLY, client_id)
        raise NotImplementedError("This method should be implemented in subclasses to return training data for the quantizer.")

    def build_quantizer(self, x: torch.Tensor, si: list[torch.Tensor]) -> WZQuantizerCancer:
        self.quantizer = quantizer = self.quantizer_class(c_cfg=self.c_cfg, si_size=len(si))
        self.quantizer.train_model(x, si)
        gc.collect()
        torch.cuda.empty_cache()
        return quantizer

    def create_encode_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        assert delta_vec.dtype == torch.float32 and delta_vec.device == torch.device("cpu")
        assert self.quantizer is not None, f"Missing quantizer for client {record.client_id}."

        bins, prep_metadata = self.quantizer.encoding_process(delta_vec)

        encoder_decoder_state = [self.quantizer.coding_model.encoder.state_dict(), 
                                 self.quantizer.coding_model.decoder.state_dict()]
        record.encoder_size = _compressed_size(encoder_decoder_state[0])
        record.decoder_size = _compressed_size(encoder_decoder_state[1])
        record.meta_data_size = _compressed_size(prep_metadata)

        payload = {
            'payload_content': (bins, prep_metadata),
            'quantizer_state': encoder_decoder_state[0]
            }
        return payload
    
    def encode(self, delta_vec: torch.Tensor, record: CancerRecord) -> bytes:
        payload = self.create_encode_payload(delta_vec, record)
        return compress_data_list(payload)

    def decode(self, payload: bytes, record: CancerRecord) -> torch.Tensor:
        payload_dict = decompress_data_list(payload)
        reconst = self.quantizer.decoding_process(payload_dict["payload_content"])
        return reconst
    

class F_RoundCodec(_WZRoundCodec):
    round_name: ClassVar[str] = "F"
    can_decode_where: Access
    frozen_si: list[torch.Tensor]

    def __init__(self, cfg_options, round_name_full,
                 f_quant: WZQuantizerCancer, f_si: list[torch.Tensor], h_access: Access) -> None:
        super().__init__(cfg_options, round_name_full)
        self.quantizer = f_quant
        self.frozen_si = f_si
        self.can_decode_where = h_access

    def build_quantizer(self, x: torch.Tensor, si: list[torch.Tensor]) -> WZQuantizerCancer:
        raise NotImplementedError("F_RoundCodec does not support building a new quantizer. Use the frozen quantizer instead.")
    
class P_RoundCodec(_WZRoundCodec):
    round_name: ClassVar[str] = "P"
    can_decode_where: Access = Access.TEMPORAL_TOO
    frozen_si: list[torch.Tensor]

    def build_quantizer(self, x: torch.Tensor, si: list[torch.Tensor]) -> WZQuantizerCancer:
        assert x is None and si is None
        self.quantizer = quantizer = self.quantizer_class(c_cfg=self.c_cfg, si_size=len(si))

        weight_path = self.c_cfg.pretrain_pth_dir + f'bpp{self.c_cfg.bins_per_plane}_np{self.c_cfg.num_planes}_pretrained_wzq_rnn.pth'
        quantizer.coding_model.load_state_dict(torch.load(weight_path), strict=False)
        quantizer.side_info_list_used = []
    

class R_RoundCodec(_WZRoundCodec):
    round_name: ClassVar[str] = "R"
    can_decode_where: Access = Access.SERVER_ONLY
    frozen_si: list[torch.Tensor]

    def get_training_data(self, given_history: ReconstructionHistory, client_id: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        history = given_history.view(Access.SERVER_ONLY, client_id)
        x_entry = history[client_id][-1]
        assert x_entry.client_id == client_id
        assert x_entry.access == Access.SERVER_ONLY
        train_si = [entry.tensor for c_h in history.values() for entry in c_h if entry != x_entry]
        return x_entry.tensor, train_si
    

class S_RoundCodec(_WZRoundCodec):
    round_name: ClassVar[str] = "S"
    can_decode_where: Access = Access.SERVER_ONLY
    frozen_si: list[torch.Tensor]
    si: list[torch.Tensor]
    
    def get_training_data(self, given_history: ReconstructionHistory, 
                          client_id: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
        history = given_history.view(Access.SERVER_ONLY, client_id)
        train_si = [entry.tensor for c_h in history.values() for entry in c_h]
        x_tensor = torch.stack([c_h[-1].tensor for c_h in history.values()])
        x_tensor = x_tensor[torch.randint(x_tensor.shape[0], (), device=x_tensor.device)]

        return x_tensor, train_si

    def create_encode_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        # sample_idx = ?
        # bins_of_sampled = ?
        # somehow add a super compressed (using sw) version of the sent values to the record
        #   it has to appear in the prior metric
        #   it has to also appear in the data sent but sw used on it
        #   probably since prior compute is in encode function, it should be handled there
        # train on a sample of delta_vec (only the head of the encoder and the entire decoder)

        # include the head of the encoder in the payload

        # use the trained quantizer to encode the remaining delta_vec and send it to the server but not more probably using the super

        # account for the fact that the prior from the sampled part and the rest have to be combined in the prior metric

        return payload

    def encode(self, delta_vec: torch.Tensor, record: CancerRecord) -> bytes:
        return super().encode(delta_vec, record)
    

class T_RoundCodec(_WZRoundCodec):
    where_quantizer_trained: Literal["server", "worker"] = "worker"
    can_decode_where: ClassVar[Access] = Access.TEMPORAL_TOO
    round_name: ClassVar[str] = "T"

    si: list[torch.Tensor]

    def get_training_data(self, given_history: ReconstructionHistory, client_id: int) -> tuple[torch.Tensor|None, list[torch.Tensor]]:
        si_entry = given_history.view(Access.TEMPORAL_TOO, client_id)[client_id]
        assert si_entry[-1].client_id == client_id
        assert si_entry[-1].access == Access.TEMPORAL_TOO
        train_si = [entry.tensor for entry in si_entry]
        return None, train_si

    def build_quantizer(self, x: torch.Tensor, si: list[torch.Tensor]) -> WZQuantizerCancer:
        self.quantizer = quantizer = self.quantizer_class(c_cfg=self.c_cfg, si_size=len(si))
        return quantizer

    def create_encode_payload(self, delta_vec: torch.Tensor, record: CancerRecord) -> dict[str, Any]:
        self.quantizer.train_model(delta_vec, self.si)
        gc.collect()
        torch.cuda.empty_cache()
        payload = super().create_encode_payload(delta_vec, record)
        payload['quantizer_state'] = self.quantizer.coding_model.decoder.state_dict()
        return payload


class _WZProtocol(BaseProtocol):
    max_per_client_recons_history = 5

    warmup_round_codecs: ClassVar[tuple[str, ...]]
    routine_round_codecs: ClassVar[tuple[str, ...]]
    protocol_name: ClassVar[str]

    last_frozen_state: dict[int, dict[str, Any]] = {}

    def create_round_codec(self, round_id: int, client_id: int) -> _WZRoundCodec:
        rc_class, parsed, round_name_full = self._get_curr_round_codec_name(round_id)
        if rc_class is F_RoundCodec:
            codec = F_RoundCodec(parsed.options, round_name_full, *self.last_frozen_state.get(client_id))
            assert isinstance(codec, F_RoundCodec)
            return codec

        codec = rc_class(parsed.options, round_name_full)
        assert isinstance(codec, _WZRoundCodec)
        
        t_x, t_si = codec.get_training_data(self._recons_history, client_id)
        assert t_x is None or isinstance(codec, _WZRoundCodec)

        quantizer = codec.build_quantizer(t_x, t_si)

        self.last_frozen_state[client_id] = {
            'f_quant': quantizer,
            'f_si': t_si,
            'h_access': codec.can_decode_where
        }

        return codec
