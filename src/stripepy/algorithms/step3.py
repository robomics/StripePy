# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import time
from typing import Callable, Optional, Tuple

import numpy as np
import structlog
from numpy.typing import NDArray

from stripepy.algorithms.finders import (
    find_horizontal_intervals_of_interest,
    find_vertical_intervals_of_interest,
)
from stripepy.data_structures import Result, SparseMatrix
from stripepy.utils import pretty_format_elapsed_time


def run(
    result: Result,
    matrix: Optional[SparseMatrix],
    resolution: int,
    max_stripe_height: int,
    max_stripe_width: int,
    loc_pers_min: float,
    loc_trend_min: float,
    location: str,
    map_: Callable = map,
    logger=None,
) -> Tuple[str, Result]:
    """
    Compute the width and height of each candidate stripe.

    Parameters
    ----------
    result : Result
        the Result object produced by step_2()
    matrix: Optional[SparseMatrix]
        matrix with the interactions to be processed.
        When set to None, the matrix will be fetched from the global shared state.
    resolution: int
        resolution of the matrix in base-pairs.
    max_stripe_height: int
        maximum stripe height in base-pairs.
    max_stripe_width: int
        maximum stripe width in base-pairs.
    loc_pers_min: float
        threshold value between 0 and 1 to find peaks in signal in a horizontal
        domain while estimating the height of a stripe
    loc_trend_min: float
        threshold value between 0 and 1 to estimate the height of a stripe.
        The higher this value, the shorter the stripe
    location: str
        matrix location (should be "lower" or "upper")
    map_: Callable
        a callable that behaves like the built-in map function
    logger:
        logger

    Returns
    -------
    str
        location (same as the location given as input).
    Result
        A copy of the Result object given as input with the stripe widths
        (left_bound and right_bound) and heights (top_bound and bottom_bound) set.
    """
    assert resolution > 0
    assert max_stripe_height > 0
    assert max_stripe_width > 0
    assert 0 <= loc_pers_min <= 1
    assert 0 <= loc_trend_min <= 1
    assert location in {"lower", "upper"}

    if logger is None:
        logger = structlog.get_logger().bind(chrom=result.chrom[0], step=(3,))

    logger = logger.bind(location="LT" if location == "lower" else "UT")

    if len(result.get("stripes", location)) == 0:
        logger.bind(step=(3,)).warning("no candidates found by step 2: returning immediately!")
        return location, result

    start_time = time.time()

    persistent_min_points = result.get("persistent_minimum_points", location)
    persistent_max_points = result.get("persistent_maximum_points", location)
    pseudodistribution = result.get("pseudodistribution", location)

    logger.bind(step=(3, 1)).info("estimating candidate stripe widths")

    # Ensure that each maximum point is between two minimum points
    persistent_min_points_bounded = _complement_persistent_minimum_points(
        pseudodistribution, persistent_min_points, persistent_max_points
    )

    # DataFrame with the left and right boundaries for each seed site
    horizontal_domains = find_horizontal_intervals_of_interest(
        pseudodistribution=pseudodistribution,
        seed_sites=persistent_max_points,
        seed_site_bounds=persistent_min_points_bounded,
        max_width=int(max_stripe_width / (2 * resolution)) + 1,
        logger=logger,
    )

    domain_widths = horizontal_domains["right_bound"] - horizontal_domains["left_bound"]

    logger.bind(step=(3, 1)).info(
        "width estimation of %d stripes took %s (mean=%.0f kbp; std=%.0f kbp)",
        len(domain_widths),
        pretty_format_elapsed_time(start_time),
        domain_widths.mean() * resolution / 1000,
        domain_widths.std() * resolution / 1000,
    )

    stripes = result.get("stripes", location)

    logger.bind(step=(3, 2)).info("updating candidate stripes with width information")
    horizontal_domains.apply(
        lambda seed: stripes[seed.name].set_horizontal_bounds(seed["left_bound"], seed["right_bound"]),
        axis="columns",
    )

    start_time = time.time()

    logger.bind(step=(3, 3)).info("estimating candidate stripe heights")
    vertical_domains = find_vertical_intervals_of_interest(
        matrix=matrix,
        seed_sites=persistent_max_points,
        horizontal_domains=horizontal_domains,
        max_height=int(max_stripe_height / resolution),
        threshold_cut=loc_trend_min,
        min_persistence=loc_pers_min,
        location=location,
        map_=map_,
        logger=logger,
    )

    domain_heights = (vertical_domains["top_bound"] - vertical_domains["bottom_bound"]).abs()

    logger.bind(step=(3, 3)).info(
        "height estimation of %d stripes tool %s (mean=%.0f kbp; std=%.0f kbp)",
        len(domain_heights),
        pretty_format_elapsed_time(start_time),
        domain_heights.mean() * resolution / 1000,
        domain_heights.std() * resolution / 1000,
    )

    logger.bind(step=(3, 4)).info("updating candidate stripes with height information")
    vertical_domains[["top_bound", "bottom_bound"]].apply(
        lambda seed: stripes[seed.name].set_vertical_bounds(seed["top_bound"], seed["bottom_bound"]),
        axis="columns",
    )

    return location, result


def _complement_persistent_minimum_points(
    pseudodistribution: NDArray[float],
    persistent_minimum_points: NDArray[int],
    persistent_maximum_points: NDArray[int],
) -> NDArray[int]:
    """
    Find the minimum values associated with the first and last persistent maximum points

    Returns
    -------
    NDArray[int]
        The persistent minimum point given as input with the new bounds added,
        i.e. the original vector with the minimum point to the left of the first max point prepended,
        and the minimum point to the right of the last maximum point appended
    """
    assert len(persistent_maximum_points) > 0
    assert len(pseudodistribution) != 0

    i0 = persistent_maximum_points[0]
    i1 = persistent_maximum_points[-1]

    if i0 != 0:
        # Get the coordinate of the minimum point to the left of the first max point
        left_bound = np.argmin(pseudodistribution[:i0])
    else:
        left_bound = 0

    if i1 != len(pseudodistribution):
        # Get the coordinate of the minimum point to the right of the last max point
        right_bound = i1 + np.argmin(pseudodistribution[i1:])
    else:
        right_bound = len(pseudodistribution)

    # We need to check that the list of minimum points are not empty, otherwise np.concatenate will create an array with dtype=float
    if len(persistent_minimum_points) == 0:
        return np.array([left_bound, right_bound], dtype=int)

    # Persistent minimum points are always between two maxima, thus the left and right bounds should never be the same as the first and last min points
    assert left_bound != persistent_minimum_points[0]
    assert right_bound != persistent_minimum_points[-1]

    return np.concatenate([[left_bound], persistent_minimum_points, [right_bound]], dtype=int)
