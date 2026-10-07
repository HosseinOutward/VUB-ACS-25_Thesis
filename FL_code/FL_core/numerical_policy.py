"""Explicit numerical execution policy shared by FL generation and learned codecs."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Literal

from pydantic import BaseModel, ConfigDict
import torch


class NumericalPolicy(BaseModel):
    """Backend settings selected by an execution boundary and recorded by artifact producers.

    The default preserves CUDA FL's TF32-enabled policy. Codecs may explicitly select
    ``highest`` and disable cuDNN TF32 independently of automatic mixed precision.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    float32_matmul_precision: Literal["highest", "high", "medium"] = "high"
    cudnn_allow_tf32: bool = True
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = False
    matmul_allow_fp16_reduced_precision_reduction: bool = True
    matmul_allow_bf16_reduced_precision_reduction: bool = True

    @classmethod
    def current(cls) -> NumericalPolicy:
        """Capture every backend setting this policy controls, without changing the process."""
        return cls(
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_allow_fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            matmul_allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )

    def _apply(self) -> None:
        # allow_tf32 aliases this setting; assigning it afterwards would turn medium into high.
        torch.set_float32_matmul_precision(self.float32_matmul_precision)
        torch.backends.cudnn.allow_tf32 = self.cudnn_allow_tf32
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = self.matmul_allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = self.matmul_allow_bf16_reduced_precision_reduction

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Establish this policy temporarily, restoring the complete caller state even on failure."""
        previous = self.current()
        try:
            self._apply()
            yield
        finally:
            previous._apply()

    def runtime(self, device: torch.device, *, mixed_precision: bool) -> dict[str, Any]:
        """Describe effective settings and hardware for cache matches and artifact provenance."""
        with self.scope():
            settings = self.current().model_dump(mode="json") | {
                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "mixed_precision": mixed_precision,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            }
        return {
            **settings, "torch_version": str(torch.__version__), "device": str(device),
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
        }
