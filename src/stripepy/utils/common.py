# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import time
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray


def sort_based_on_arg0(*vectors: Sequence) -> Tuple[NDArray]:
    """
    Sort two or more sequences of objects based on the first sequence of objects.

    Parameters
    ----------
    vectors: two or more sequences to be sorted

    Returns
    -------
    Tuple[NDArray]
        the sorted sequences as numpy.ndarray
    """
    if len(vectors) < 2:
        raise ValueError("please specify at least two sequences")

    for v in vectors[1:]:
        assert len(vectors[0]) == len(v)

    if len(vectors[0]) == 0:
        return tuple((np.array(v) for v in vectors))  # noqa

    permutation = np.argsort(vectors[0])

    return tuple((np.array(v)[permutation] for v in vectors))  # noqa


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
