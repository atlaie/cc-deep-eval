"""Pydantic models for vllm-lens steering vectors."""

from __future__ import annotations

from typing import Any, Self

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)

from vllm_lens._helpers._serialize import deserialize_tensor, serialize_tensor


class SteeringVector(BaseModel):
    """A steering vector that modifies the residual stream during inference."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    activations: torch.Tensor
    layer_indices: list[int]
    scale: float = 1.0
    norm_match: bool = False
    position_indices: list[int] | None = None

    @field_validator("activations", mode="before")
    @classmethod
    def _deserialize_activations(cls, v: Any) -> torch.Tensor:
        if isinstance(v, dict) and "data" in v:
            return deserialize_tensor(v)
        if isinstance(v, torch.Tensor):
            return v
        raise ValueError(
            f"activations must be a torch.Tensor or a base64 dict, got {type(v)}"
        )

    @field_serializer("activations")
    def _serialize_activations(self, v: torch.Tensor, _info: Any) -> dict[str, Any]:
        return serialize_tensor(v)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.activations.dim() not in (2, 3):
            raise ValueError(
                f"activations must be 2D or 3D, got {self.activations.dim()}D"
            )
        if self.activations.shape[0] != len(self.layer_indices):
            raise ValueError(
                f"activations dim 0 ({self.activations.shape[0]}) must match "
                f"len(layer_indices) ({len(self.layer_indices)})"
            )
        return self

    @property
    def layer_index_map(self) -> dict[int, int]:
        return {li: i for i, li in enumerate(self.layer_indices)}
