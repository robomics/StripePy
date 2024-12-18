# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

from typing import Sequence, Tuple

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
