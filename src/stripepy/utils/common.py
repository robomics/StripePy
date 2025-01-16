# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import decimal
import time
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as ss
from numpy.typing import NDArray


def pretty_format_elapsed_time(t0: float, t1: Optional[float] = None) -> str:
    if t1 is None:
        t1 = time.time()

    microsecond = 1.0e-6
    millisecond = 1.0e-3
    second = 1.0
    minute = 60.0
    hour = 3600.0

    delta = t1 - t0

    if delta < microsecond:
        return f"{delta * 1.0e9:.0f}ns"

    if delta < millisecond:
        return f"{delta * 1.0e6:.3f}us"

    if delta < second:
        return f"{delta * 1000:.3f}ms"

    if delta < minute:
        return f"{delta:.3f}s"

    if delta < hour:
        minutes = delta // 60
        seconds = delta - (minutes * 60)
        return f"{minutes:.0f}m:{seconds:.3f}s"

    hours = delta // 3600
    minutes = (delta - (hours * 3600)) // 60
    seconds = delta - (hours * 3600) - (minutes * 60)
    return f"{hours:.0f}h:{minutes:.0f}m:{seconds:.3f}s"


def truncate_np(v: NDArray[float], places: int) -> NDArray[float]:
    """
    Truncate a numpy array to the given number of decimal places.
    Implementation based on https://stackoverflow.com/a/28323804

    Parameters
    ----------
    v: NDArray[float]
        the numpy array to be truncated
    places: int
        the number of decimal places to truncate to

    Returns
    -------
    NDArray[float]
        numpy array with truncated values
    """
    assert places >= 0

    if places == 0:
        return v.round(0)

    with decimal.localcontext() as context:
        context.rounding = decimal.ROUND_DOWN
        exponent = decimal.Decimal(str(10**-places))
        return np.array([float(decimal.Decimal(str(n)).quantize(exponent)) for n in v], dtype=float)


def split_df(df: pd.DataFrame, num_chunks: int) -> List[pd.DataFrame]:
    assert num_chunks > 0

    offsets = np.round(np.linspace(0, len(df), num_chunks + 1)).astype(int)
    chunks = []
    for i in range(1, len(offsets)):
        i0 = offsets[i - 1]
        i1 = offsets[i]
        chunks.append(df.iloc[i0:i1])

    # assert len(chunks) == num_chunks
    # assert sum((len(dff) for dff in chunks)) == len(df)

    return chunks


def _complement_indices(indices: Sequence[int], max_idx: int, min_idx: int = 0) -> NDArray[int]:
    return np.setdiff1d(np.arange(min_idx, max_idx), indices)


def zero_rows(matrix: ss.csr_matrix, rows_whitelist: Sequence[int]) -> ss.csr_matrix:
    """
    https://stackoverflow.com/a/43114513
    """
    indices = _complement_indices(rows_whitelist, matrix.shape[0])
    diag = ss.eye(matrix.shape[0]).tolil()
    diag[indices, indices] = 0
    return diag.dot(matrix).tocsr()


def zero_columns(matrix: ss.csr_matrix, columns_whitelist: Sequence[int]) -> ss.csc_matrix:
    """
    https://stackoverflow.com/a/43114513
    """
    indices = _complement_indices(columns_whitelist, matrix.shape[0])
    diag = ss.eye(matrix.shape[0]).tolil()
    diag[indices, indices] = 0
    return matrix.dot(diag).tocsc()


def _import_matplotlib():
    """
    Helper function to import matplotlib.
    """
    try:
        import matplotlib

        return matplotlib
    except ImportError as e:
        raise ImportError(
            "unable to import matplotlib: to enable matplotlib support, please install StripePy with: pip install 'stripepy-hic[all]'"
        ) from e


def _import_pyplot():
    """
    Helper function to import matplotlib.pyplot.
    """
    _import_matplotlib()
    import matplotlib.pyplot as plt

    return plt


class _DummyPyplot(object):
    """
    class to mock common types from matplotlib.pyplot.
    """

    def __init__(self):
        self.Figure = None
        self.Axes = None
