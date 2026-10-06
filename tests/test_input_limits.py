from __future__ import annotations

import io

import numpy as np
import pytest

from tsdf_fusion216.input_validation import load_depth_npz


def test_npz_object_array_is_rejected() -> None:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        depths=np.array([[np.array([1.0])]], dtype=object),
        K=np.eye(3),
        Tcw=np.eye(4)[None],
        allow_pickle=True,
    )
    with pytest.raises(ValueError):
        load_depth_npz(buffer.getvalue())
