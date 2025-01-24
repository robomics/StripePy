# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import itertools
import time
from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as ss
import structlog

from stripepy.utils.common import pretty_format_elapsed_time
from stripepy.utils.multiprocess_sparse_matrix import SparseMatrix, get_shared_state
from stripepy.utils.persistence1d import PersistenceTable
from stripepy.utils.regressions import compute_wQISA_predictions


def find_horizontal_domain(
    profile: npt.NDArray[float],
    params: Tuple[int, int, int, float],
    max_width: int = 1e9,
) -> Tuple[int, int]:
    """
    Returns
    -------
    Tuple[int, int]
        the left and right coordinates of the horizontal domain
    """

    # Unpacking:
    MP, L_mP, R_mP, max_profile_value = params

    # Left and sides of candidate:
    L_interval = np.flip(profile[L_mP : MP + 1])
    R_interval = profile[MP : R_mP + 1]

    # LEFT INTERVAL
    L_interval_shifted = np.append(L_interval[1:], [max_profile_value + 1], axis=0)
    L_bound = np.where(L_interval - L_interval_shifted < 0)[0][0] + 1
    # L_interval_restr = L_interval[:L_bound]
    # L_interval_shifted_restr = L_interval_shifted[:L_bound]
    # L_bound = np.argmax(L_interval_restr - L_interval_shifted_restr) + 1
    L_bound = np.minimum(L_bound, max_width)

    # RIGHT INTERVAL
    R_interval_shifted = np.append(R_interval[1:], [max_profile_value + 1], axis=0)
    R_bound = np.where(R_interval - R_interval_shifted < 0)[0][0] + 1
    # R_interval_restr = R_interval[:R_bound]
    # R_interval_shifted_restr = R_interval_shifted[:R_bound]
    # R_bound = np.argmax(R_interval_restr - R_interval_shifted_restr) + 1
    R_bound = np.minimum(R_bound, max_width)

    return max(MP - L_bound, 0), min(MP + R_bound, len(profile))


def _extract_standardized_local_1d_pseudodistribution(
    matrix: SparseMatrix,
    seed_site: int,
    left_bound: int,
    right_bound: int,
    max_height: int,
    location: str,
) -> npt.NDArray[float]:
    # assert left_bound <= seed_site <= right_bound

    if location == "lower":
        i0, i1 = seed_site, min(seed_site + max_height, matrix.shape[0])
    else:
        i0, i1 = max(seed_site - max_height, 0), seed_site

    j0, j1 = left_bound, right_bound
    if isinstance(matrix, ss.csc_matrix):
        submatrix = matrix[:, j0:j1].tocsr()[i0:i1, :].tocsc()
    elif isinstance(matrix, ss.csr_matrix):
        submatrix = matrix[i0:i1, :].tocsc()[:, j0:j1]
    else:
        return _extract_standardized_local_1d_pseudodistribution(
            matrix=matrix.tocsr(),
            seed_site=seed_site,
            left_bound=left_bound,
            right_bound=right_bound,
            max_height=max_height,
            location=location,
        )

    y = submatrix.sum(axis=1)
    if isinstance(y, np.matrix):
        y = y.A1

    y /= y.max()
    if location == "lower":
        return y

    # assert location == "upper"
    return np.flip(y)


def _find_v_domain_helper(
    profile: npt.NDArray[float], threshold_cut: float, min_persistence: Optional[float]
) -> Tuple[int, Optional[npt.NDArray[int]]]:
    if min_persistence is None:
        max_points = tuple()
    else:
        max_points = (
            PersistenceTable.calculate_persistence(profile, min_persistence=min_persistence)
            .max.sort_values(kind="stable")
            .index.to_numpy()
        )

    if len(max_points) > 1:
        return max_points.max(), max_points[:-1]  # drop global maximum

    approximated_profile = compute_wQISA_predictions(profile, 5)
    candida_bound = int(np.argmax(approximated_profile < threshold_cut))
    if approximated_profile[candida_bound] >= threshold_cut:
        candida_bound = len(profile) - 1

    return candida_bound, None


def _find_lower_v_domain(
    coords: Tuple[int, int, int],
    matrix: Optional[SparseMatrix],
    threshold_cut: float,
    max_height: int,
    min_persistence: float,
    return_maxima: bool,
    location: str,
) -> Dict[str, Any]:
    seed_site, left_bound, right_bound = coords
    # assert left_bound <= seed_site <= right_bound

    if matrix is None:
        matrix = get_shared_state(location).get()

    profile = _extract_standardized_local_1d_pseudodistribution(
        matrix, seed_site, left_bound, right_bound, max_height, "lower"
    )
    candidate_bound, local_maxima = _find_v_domain_helper(profile, threshold_cut, min_persistence)

    res = {"top_bound": seed_site, "bottom_bound": seed_site + candidate_bound}

    if not return_maxima or local_maxima is None:
        return res

    local_maxima += seed_site
    res["maxima"] = local_maxima
    return res


def _find_upper_v_domain(
    coords: Tuple[int, int, int],
    matrix: Optional[SparseMatrix],
    threshold_cut: float,
    max_height: int,
    min_persistence: float,
    return_maxima: bool,
    location: str,
) -> Dict[str, Any]:
    seed_site, left_bound, right_bound = coords
    # assert left_bound <= seed_site <= right_bound

    if matrix is None:
        matrix = get_shared_state(location).get()

    profile = _extract_standardized_local_1d_pseudodistribution(
        matrix, seed_site, left_bound, right_bound, max_height, "upper"
    )
    candidate_bound, local_maxima = _find_v_domain_helper(profile, threshold_cut, min_persistence)

    res = {"top_bound": seed_site - candidate_bound, "bottom_bound": seed_site}

    if not return_maxima or local_maxima is None:
        return res

    local_maxima *= -1
    local_maxima += seed_site

    res["maxima"] = local_maxima
    return res


def find_HIoIs(
    pseudodistribution: npt.NDArray[float],
    seed_sites: npt.NDArray[int],
    seed_site_bounds: npt.NDArray[int],
    max_width: int,
    logger=None,
) -> pd.DataFrame:
    """
    :param pseudodistribution:  1D array representing a uniformly-sample scalar function works
    :param seed_sites:          maximum values in the pseudo-distribution (i.e., genomic coordinates hosting linear
                                patterns)
    :param seed_site_bounds:    for the i-th entry of seed_sites:
                                (*) seed_site_bounds[i] is the left boundary
                                (*) seed_site_bounds[i+1] is the right boundary
    :param max_width:           maximum width allowed
    :return:
    HIoIs                       a pd.DataFrame the list of left and right boundary for each seed site
    """
    assert len(seed_site_bounds) == len(seed_sites) + 1

    t0 = time.time()
    if logger is None:
        logger = structlog.get_logger()

    max_pseudodistribution_value = pseudodistribution.max()
    params = (
        (seed_site, seed_site_bounds[num_MP], seed_site_bounds[num_MP + 1], max_pseudodistribution_value)
        for num_MP, seed_site in enumerate(seed_sites)
    )

    tasks = map(partial(find_horizontal_domain, pseudodistribution, max_width=max_width), params)
    # This efficiently constructs a 2D numpy with shape (N, 2) from a list of 2-element tuples, where N is the number of seed sites.
    # The first and second columns contains the left and right boundaries of the horizontal domains, respectively.
    HIoIs = np.fromiter(itertools.chain.from_iterable(tasks), count=2 * len(seed_sites), dtype=int).reshape(-1, 2)

    # Handle possible overlapping intervals by ensuring that the
    # left bound of interval i + 1 is always greater or equal than the right bound of interval i
    HIoIs[1:, 0] = np.maximum(HIoIs[1:, 0], HIoIs[:-1, 1])

    df = pd.DataFrame(data=HIoIs, columns=["left_bound", "right_bound"])

    logger.debug("find_HIoIs took %s", pretty_format_elapsed_time(t0))

    return df


def find_VIoIs(
    matrix: Optional[SparseMatrix],
    seed_sites: npt.NDArray[int],
    horizontal_domains: pd.DataFrame,
    max_height: int,
    threshold_cut: float,
    min_persistence: float,
    location: str,
    return_maxima: bool = False,
    map_=map,
    logger=None,
) -> pd.DataFrame:
    assert len(seed_sites) > 0
    assert len(seed_sites) == len(horizontal_domains)

    t0 = time.time()
    if logger is None:
        logger = structlog.get_logger()

    df = pd.concat([pd.DataFrame({"seed_site": seed_sites}), horizontal_domains], axis="columns")

    if location == "lower":
        finder = _find_lower_v_domain
    elif location == "upper":
        finder = _find_upper_v_domain
    else:
        raise ValueError("where should be lower or upper")

    tasks = map_(
        partial(
            finder,
            matrix=matrix,
            threshold_cut=threshold_cut,
            max_height=max_height,
            min_persistence=min_persistence,
            return_maxima=return_maxima,
            location=location,
        ),
        ((seed, lb, rb) for seed, lb, rb in df[["seed_site", "left_bound", "right_bound"]].itertuples(index=False)),
    )

    df = pd.DataFrame.from_records(
        data=tasks,
        nrows=len(seed_sites),
    )

    assert len(df) == len(seed_sites)

    logger.debug("find_VIoIs took %s", pretty_format_elapsed_time(t0))

    return df
