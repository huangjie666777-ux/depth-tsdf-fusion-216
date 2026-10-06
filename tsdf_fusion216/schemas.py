"""Request parameter models for the fusion endpoint."""

from __future__ import annotations

import json
import math

from pydantic import BaseModel, Field, field_validator

from .errors import InputValidationError

MIN_DIM = 2
MAX_DIM = 64


class FusionParams(BaseModel):
    """Volume parameters; all lengths are in metres."""

    origin: tuple[float, float, float] = Field(
        ..., description="World-space minimum corner of the volume."
    )
    dims: tuple[int, int, int] = Field(
        ..., description="Voxel counts along (x, y, z), each in [2, 64]."
    )
    voxel_size: float = Field(..., gt=0.0, description="Voxel edge length in metres.")
    trunc: float = Field(..., gt=0.0, description="Truncation distance in metres.")

    @field_validator("origin")
    @classmethod
    def _origin_finite(cls, value):
        if not all(math.isfinite(v) for v in value):
            raise ValueError("origin components must be finite")
        return value

    @field_validator("dims")
    @classmethod
    def _dims_range(cls, value):
        for d in value:
            if d < MIN_DIM or d > MAX_DIM:
                raise ValueError(f"dims must be within [{MIN_DIM}, {MAX_DIM}]")
        return value

    @field_validator("voxel_size", "trunc")
    @classmethod
    def _positive_finite(cls, value):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("must be a positive finite number")
        return value


def parse_params(raw: str) -> FusionParams:
    """Parse the JSON params form field into a validated model."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InputValidationError(f"params is not valid JSON: {exc}") from exc
    try:
        return FusionParams.model_validate(payload)
    except Exception as exc:
        raise InputValidationError(f"invalid params: {exc}") from exc
