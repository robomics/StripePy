# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from numpy.typing import NDArray

from stripepy.algorithms.regressions import compute_wQISA_predictions
from stripepy.data_structures import (
    Persistence1DTable,
    Result,
    SparseMatrix,
    Stripe,
    get_shared_state,
)
from stripepy.utils import truncate_np


def run(
    chrom_name: str,
    chrom_size: int,
    matrix: Optional[SparseMatrix],
    min_persistence: float,
    location: str,
    logger=None,
) -> Tuple[str, Result]:
    """
    Compute the global 1D pseudo-distribution and identify the coordinates of the candidate stripe seeds.

    Parameters
    ----------
    chrom_name: str
        name of the chromosome to be processed.
    chrom_size: int
        size of the chromosome to be processed.
    matrix: Optional[SparseMatrix]
        matrix with the interactions to be processed.
        When set to None, the matrix will be fetched from the global shared state.
    min_persistence: float
        minimum persistence threshold.
    location: str
        matrix location (should be "lower" or "upper")
    logger:
        logger

    Returns
    -------
    str
        location (same as the location given as input).
    ResultFile
        Result object with the following attributes set:
        - chromosome
        - pseudodistribution
        - min_persistence
        - all_minimum_points
        - all_maximum_points
        - persistence_of_all_minimum_points
        - persistence_of_all_maximum_points
        - stripes (seed and top persistence values only)
    """
    assert chrom_size > 0
    assert 0 <= min_persistence <= 1
    assert location in {"lower", "upper"}

    if logger is None:
        logger = structlog.get_logger().bind(chrom=chrom_name, step=(2,))

    logger = logger.bind(location="LT" if location == "lower" else "UT")

    if matrix is None:
        # Fetch matrix from the shared global state
        matrix = get_shared_state(location).get()

    # Initialize result object
    result = Result(chrom_name, chrom_size)

    logger.bind(step=(2, 1, 0)).info("computing global 1D pseudo-distribution")
    pseudodistribution = _compute_global_pseudodistribution(matrix, smooth=True)

    result.set_min_persistence(min_persistence)
    result.set("pseudodistribution", pseudodistribution, location)

    logger.bind(step=(2, 2, 0)).info("detection of persistent maxima and corresponding minima")

    logger.bind(step=(2, 2, 1)).info("computing persistence")
    persistence = Persistence1DTable.calculate_persistence(pseudodistribution, min_persistence=0, sort_by="persistence")
    result.set("all_minimum_points", persistence.min.index.to_numpy(copy=True), location)
    result.set("all_maximum_points", persistence.max.index.to_numpy(copy=True), location)
    result.set("persistence_of_all_minimum_points", persistence.min.to_numpy(copy=True), location)
    result.set("persistence_of_all_maximum_points", persistence.max.to_numpy(copy=True), location)

    logger.bind(step=(2, 2, 2)).info("filtering low persistence values")
    persistence.filter(min_persistence, method="greater")
    persistence.sort(by="position")

    logger.bind(step=(2, 2, 3)).info("removing seeds overlapping sparse regions")
    data = _filter_extrema_by_sparseness(
        matrix=matrix,
        min_points=persistence.min,
        max_points=persistence.max,
        logger=logger,
    )
    persistence = Persistence1DTable(pers_of_min_points=data[0], pers_of_max_points=data[1], level_sets="upper")
    result.set("persistent_minimum_points", persistence.min.index.to_numpy(), location)
    result.set("persistent_maximum_points", persistence.max.index.to_numpy(), location)
    result.set("persistence_of_minimum_points", persistence.min.to_numpy(), location)
    result.set("persistence_of_maximum_points", persistence.max.to_numpy(), location)

    if len(persistence.max) == 0:
        logger.bind(step=(2, 2, 3)).warning("no seed survived filtering by sparseness")
        result.set("stripes", [], location)
        return location, result

    logger.bind(step=(2, 3, 1)).info("generating the list of candidate stripes")
    where = f"{location}_triangular"
    stripes = [Stripe(seed=x, top_pers=pers, where=where) for x, pers in persistence.max.items()]  # noqa
    logger.bind(step=(2, 3, 1)).info("identified %d candidate stripes", len(stripes))
    result.set("stripes", stripes, location)

    return location, result


def _compute_global_pseudodistribution(
    matrix: SparseMatrix,
    smooth: bool = True,
    decimal_places: int = 10,
) -> NDArray[float]:
    """
    Given a sparse matrix T, marginalize it, scale the marginal so that maximum is 1, and then smooth it.

    Parameters
    ----------
    matrix: SparseMatrix
        the sparse matrix to be processed
    smooth: bool
        if set to True, smoothing is applied to the pseudo-distribution (default value is True)
    decimal_places: int
        the number of decimal places to truncate the pseudo-distribution to.
        Pass -1 to not truncate the pseudo-distribution values

    Returns
    -------
    NDArray[float]
        a vector with the re-scaled and smoothed marginals.
    """

    pseudo_dist = matrix.sum(axis=0)  # marginalization
    if isinstance(pseudo_dist, np.matrix):
        pseudo_dist = pseudo_dist.A1

    pseudo_dist /= pseudo_dist.max()  # scaling
    if smooth:
        # smoothing
        pseudo_dist = np.maximum(
            compute_wQISA_predictions(pseudo_dist, 11),  # noqa
            pseudo_dist,
        )

    if decimal_places >= 0:
        # We need to truncate FP numbers to ensure that later steps generate consistent results
        # even in the presence to very minor numeric differences on different platforms.
        return truncate_np(pseudo_dist, decimal_places)

    return pseudo_dist


def _check_neighborhood(
    values: NDArray[float],
    min_value: float = 0.1,
    neighborhood_size: int = 10,
    threshold_percentage: float = 0.85,
) -> NDArray[bool]:
    """
    Given a 1D profile, it compute a mask of False values and True values where:
    a False represents a bin surrounded by a sparse region and a True represents
    a bin surrounded by a dense region.

    Parameters
    ----------
    values : NDArray[float]
        1D array representing a uniformly-sample scalar function works
    min_value: float
        threshold used to find sparse bins.
    neighborhood_size: int
        for each bin, it is used as neighborhood radius: it serves to determine a
        neighborhood which is used to establish whether the bin is in a sparse
        region.
    threshold_percentage: float
        if, for the current neighborhood, the fraction of bins with values below min_value
        is above threshold_percentage, then the region is sparse.

    Returns
    -------
    NDArray[bool]
        A 1D array of booleans, having the same length of the array named values:
        - False values represent bins surrounded by sparse regions.
        - True values represent bins surrounded by dense regions.
    # TODO rea1991: change neighborhood size from "matrix" to "genomic" (eg, default of 1 Mb)
    """

    assert 0 <= min_value
    assert 0 < neighborhood_size
    assert 0 <= threshold_percentage <= 1

    if len(values) < neighborhood_size * 2:
        # Not enough values, return a mask with all entries set to False
        return np.full_like(values, False, dtype=bool)

    # Initialize boolean mask to True, except for the first and last neighborhood_size elements
    mask = np.full_like(values, True, dtype=bool)
    mask[:neighborhood_size] = False
    mask[-neighborhood_size:] = False

    # Loop over each value that has a sufficient number of neighbor values to its left and right
    for i in range(neighborhood_size, len(values) - neighborhood_size):
        neighborhood = values[i - neighborhood_size : i + neighborhood_size + 1]
        ratio_above_min_value = (neighborhood >= min_value).sum() / len(neighborhood)

        if ratio_above_min_value < threshold_percentage:
            mask[i] = False
    return mask


def _filter_extrema_by_sparseness(
    matrix: SparseMatrix,
    min_points: pd.Series,
    max_points: pd.Series,
    logger=None,
) -> Tuple[pd.Series, pd.Series]:
    """
    This function filters out extrema surrounded by sparse regions.
    It marginalizes and smooths an input sparse matrix, then applies
    the function _check_neighborhood to determine bins surrounded by
    sparse regions. Finally, input maxima and minimum points are filtered
    to remove extrema surrounded by sparse regions.

    Parameters
    ----------
    matrix : SparseMatrix
        sparse matrix, which can be a contact map (but this is not
        strictly necessary).
    min_points: pd.Series
        1D array containing the minimum points.
    max_points: pd.Series
        1D array containing the maximum points.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        A tuple containing:
        - Minimum points that survived the filtering.
        - Maximum points that survived the filtering.
    """

    # The following asserts are disabled for performance reasons
    # assert matrix.shape[0] == matrix.shape[1]
    # assert len(min_points) + 1 == len(max_points)
    # assert min_points.index.is_monotonic_increasing
    # assert min_points.index.max() < matrix.shape[0]
    # assert max_points.index.is_monotonic_increasing
    # assert max_points.index.max() < matrix.shape[0]

    points_to_keep = np.where(_check_neighborhood(_compute_global_pseudodistribution(matrix, smooth=False)))[0]
    mask = np.isin(max_points.index.to_numpy(), points_to_keep, assume_unique=True)
    max_points_filtered = max_points[mask]
    min_points_filtered = min_points[mask[:-1]]

    # If last maximum point was discarded, we need to remove the minimum point before it
    if len(min_points_filtered) > 0 and len(min_points_filtered) == len(max_points_filtered):
        min_points_filtered = min_points_filtered.iloc[:-1]

    if logger is None:
        logger = structlog.get_logger()

    if len(max_points_filtered) == len(max_points):
        logger.bind(step=(2, 2, 3)).info("no change in the number of seed sites")
    else:
        logger.bind(step=(2, 2, 3)).info(
            "number of seed sites reduced from %d to %d",
            len(max_points),
            len(max_points_filtered),
        )

    return min_points_filtered, max_points_filtered
