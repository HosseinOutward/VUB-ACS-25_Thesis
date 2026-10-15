"""Check exact acquired-packet extraction used by the matched final-loss report."""

from pathlib import Path

import numpy as np
import torch

from Experiments.Protocol_test.postprocess_received_final_raw_mae_pilot import sample_packet
from FL_code.FL_core.explicit_cache import content_hash
from FL_code.FL_core.utils import OutlierMetadata, PreprocessMetadata, compress_data_list
from Experiments.Protocol_test.sampled_codec import SamplePayload


def test_sample_packet_reads_embedded_symbol_bytes(tmp_path: Path) -> None:
    """The comparison extracts encoded sample symbols, not the changed full-vector payload."""
    case = tmp_path / "cases" / "variant_seed43" / "target"
    case.mkdir(parents=True)
    packet = SamplePayload(4, 2, 17, b"acquired-symbol-packet",
                           PreprocessMetadata(torch.ones((1, 2)), OutlierMetadata((), None, np.array([], dtype=bool))))
    (case / "payload.bin").write_bytes(compress_data_list({"payload_content": None,
                                                            "sample_payload": compress_data_list(packet)}))

    decoded, _ = sample_packet({"_case_path": case / "complete.json"})
    assert decoded.symbols == packet.symbols
    assert content_hash(decoded.symbols) == content_hash(packet.symbols)
