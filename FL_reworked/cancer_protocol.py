from dataclasses import dataclass
from typing import Any, Dict
import torch

from FL_reworked.codec import IdentityCodec, CompressionRecord


@dataclass
class CancerConfig:
    warmup_rnds: int = 2 # minimum 1 rounds
    realign_rnds: int = 2
    train_rnds: int = 1
    frz_rnds: int = 5


class CancerRecord(CompressionRecord):
    def __init__(self, round_id: int, client_id: int, method: str = "cancer"):
        assert method == "cancer", "CancerRecord must be used by method 'cancer'"
        super().__init__(round_id, client_id, method)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        return result


class CancerCodec(IdentityCodec):
    past_srvr_reconst_per_client:List[List[??]] = []
    past_client_reconst_per_client:List[List[??]] = []

    curr_llwzrnn_models:List[??] = []
    frozen_srvr_side_info:List[??] = []

    cancer_cfg = CancerConfig()

    def create_record(self, round_id: int, client_id: int) -> CancerRecord:
        return CancerRecord(round_id, client_id, method="cancer")

    def _compress(self, delta_vec: torch.Tensor, record: CompressionRecord) -> torch.Tensor:
        client_id = record.client_id
        round_id = record.round_id
        c_cfg = self.cancer_cfg

        long_phase_interval = c_cfg.realign_rnds + c_cfg.train_rnds + c_cfg.frz_rnds
        phase_idx = (round_id - 1 - c_cfg.warmup_rnds) % long_phase_interval

        # check if the assumed number of reconstructions exist in the lists

        # first phases
        if round_id == 0:
            # first_round_init
            # train and use a pre-trained wz-rnn without side info (empty list)
            # load the pre-trained model from disk
            pass
        elif round_id <= c_cfg.warmup_rnds+1:
            # warmup_phase
            # train and use a wz-rnn with the progressive (current) server side info list (reshaped past_srvr_reconst)
            # the input should be the last recons of related client id
            # (which is excluded from the side info list used)
            pass
        # -- routine phases below --
        elif phase_idx <= c_cfg.realign_rnds:
            # temporal_only_process
            # train and use a wz-rnn with only the client side info list (reshaped past_client_reconst) which is empty at first
            # the input is the delta_vec
            pass
        elif phase_idx <= c_cfg.realign_rnds + c_cfg.train_rnds:
            # retraining_process
            # similar to warmup_phase but we use the frozen server side info list (frozen_srvr_side_info) instead of the progressive one
            pass
        elif phase_idx <= long_phase_interval:
            # frozen_process
            # no training, only use the wz-rnn trained in the retraining_process
            pass
        else:
            raise ValueError("unified phase")


        return compressed_vec

    def _decompress(self, payload: torch.Tensor, record: CompressionRecord) -> torch.Tensor:
        # use the decoder of the wz-rnn used for compression from the _compress method
        # add the recons to the past_srvr_reconst list of the related client id
        # if its the temporal_only_process phase, also add to the past_client_reconst list for the related client id
        # avoid memory copying when using the side info lists and frozen side info list
        return recons_vec
