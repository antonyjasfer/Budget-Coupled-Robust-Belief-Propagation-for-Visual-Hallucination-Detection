"""
1D Max-plus and min-plus budget convolutions with backpointer tracking.

Given two value arrays A[0..K] and B[0..K] representing maximum (or minimum)
reachable cavity fields for budget indices 0..K:
    (A (+) B)[m] = max_{0 <= j <= m} (A[m - j] + B[j])
"""

from typing import Tuple
import numpy as np


def max_plus_convolve(
    A: np.ndarray,
    B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D max-plus convolution C = A (+) B on discrete budget indices.

    C[m] = max_{0 <= j <= m} (A[m - j] + B[j])

    Args:
        A: Array of shape (K + 1,).
        B: Array of shape (K + 1,).

    Returns:
        C: Convolved array of shape (K + 1,).
        backpointers: Array of shape (K + 1,) containing optimal j* for each m.
    """
    K = len(A) - 1
    if len(B) != K + 1:
        raise ValueError(f"Arrays must have matching length, got {len(A)} and {len(B)}")

    C = np.full(K + 1, -np.inf, dtype=np.float64)
    backpointers = np.zeros(K + 1, dtype=np.int64)

    # Compute convolution
    for m in range(K + 1):
        # j ranges from 0 to m
        j_indices = np.arange(m + 1, dtype=np.int64)
        vals = A[m - j_indices] + B[j_indices]
        best_idx = int(np.argmax(vals))
        C[m] = vals[best_idx]
        backpointers[m] = best_idx

    return C, backpointers


def min_plus_convolve(
    A: np.ndarray,
    B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D min-plus convolution C = A (+) B on discrete budget indices.

    C[m] = min_{0 <= j <= m} (A[m - j] + B[j])

    Args:
        A: Array of shape (K + 1,).
        B: Array of shape (K + 1,).

    Returns:
        C: Convolved array of shape (K + 1,).
        backpointers: Array of shape (K + 1,) containing optimal j* for each m.
    """
    K = len(A) - 1
    if len(B) != K + 1:
        raise ValueError(f"Arrays must have matching length, got {len(A)} and {len(B)}")

    C = np.full(K + 1, np.inf, dtype=np.float64)
    backpointers = np.zeros(K + 1, dtype=np.int64)

    for m in range(K + 1):
        j_indices = np.arange(m + 1, dtype=np.int64)
        vals = A[m - j_indices] + B[j_indices]
        best_idx = int(np.argmin(vals))
        C[m] = vals[best_idx]
        backpointers[m] = best_idx

    return C, backpointers
