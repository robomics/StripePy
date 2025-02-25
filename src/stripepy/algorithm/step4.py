# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import functools
from typing import List, Optional, Tuple

import numpy as np
import structlog

from stripepy.utils import stripe
from stripepy.utils.shared_sparse_matrix import SparseMatrix, get_shared_state


def run(
    stripes: List[stripe.Stripe],
    matrix: Optional[SparseMatrix],
    location: str,
    map_=map,
    logger=None,
    window: int = 3,
) -> Tuple[str, List[stripe.Stripe]]:
    """
    Compute the biodescriptors for the stripes identified by the previous steps.

    Parameters
    ----------
    stripes : List[stripe.Stripe]
        the list of stripes to be processed.
    matrix: Optional[SparseMatrix]
        matrix with the interactions to be processed.
        When set to None, the matrix will be fetched from the global shared state.
    location: str
        matrix location (should be "lower" or "upper")
    map_: Callable
        a callable that behaves like the built-in map function
    logger:
        logger
    window: int
        the window size used to compute the biodescriptors

    Returns
    -------
    str
        location (same as the location given as input).
    List[stripe.Stripe]
        a copy of the stripes given as input with their biodescriptors computed.
    """
    assert location in {"lower", "upper"}
    assert window >= 3

    if logger is None:
        logger = structlog.get_logger().bind(step=(4,))

    logger = logger.bind(location="LT" if location == "lower" else "UT")

    if len(stripes) == 0:
        logger.bind(step=(4,)).warning("no candidates found by step 2: returning immediately!")
        return location, stripes

    logger.bind(step=(4, 1)).info("computing stripe biological descriptors")

    stripes = list(
        map_(
            functools.partial(
                _step_4_helper,
                matrix=matrix,
                window=window,
                location=location,
            ),
            stripes,
        )
    )

    bad_stripe_indices = []
    for i, stripe in enumerate(stripes):
        if not np.isfinite(stripe.rel_change):
            bad_stripe_indices.append(i)

    for i in bad_stripe_indices[::-1]:
        stripes.pop(i)

    return location, stripes


def _step_4_helper(
    stripe: stripe.Stripe,  # noqa
    matrix: Optional[SparseMatrix],
    window: int,
    location: str,
) -> stripe.Stripe:
    """
    Helper function for step_4().
    Computes the biodescriptors for the given stripe.
    """
    if matrix is None:
        matrix = get_shared_state(location).get()

    stripe.compute_biodescriptors(matrix, window=window)

    return stripe
