from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asset_management_toolkit.estimation import apply_psd_policy


def test_raise_policy_rejects_indefinite_matrix() -> None:
    matrix = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        apply_psd_policy(matrix, policy="raise")


def test_clip_policy_repairs_matrix_and_preserves_labels() -> None:
    matrix = pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )
    original = matrix.copy(deep=True)

    result = apply_psd_policy(matrix, policy="clip")

    assert result.index.equals(matrix.index)
    assert result.columns.equals(matrix.columns)
    assert np.linalg.eigvalsh(result).min() >= 0.0
    pd.testing.assert_frame_equal(matrix, original)


def test_raise_policy_accepts_small_negative_eigenvalue_in_tolerance() -> None:
    matrix = pd.DataFrame(
        [[1.0, 1.0 + 1e-12], [1.0 + 1e-12, 1.0]],
        index=["a", "b"],
        columns=["a", "b"],
    )

    result = apply_psd_policy(matrix, policy="raise", tolerance=1e-10)

    pd.testing.assert_frame_equal(result, matrix)
