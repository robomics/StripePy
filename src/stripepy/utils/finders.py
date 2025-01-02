# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

from functools import partial
from typing import List, Optional, Tuple

import numpy as np

from . import TDA
from .regressions import _compute_wQISA_predictions


def find_horizontal_domain(pd, coarse_h_domain, max_width=1e9):

    # Unpacking:
    MP, L_mP, R_mP = coarse_h_domain

    # Left and sides of candidate:
    L_interval = np.flip(pd[L_mP : MP + 1])
    R_interval = pd[MP : R_mP + 1]

    # LEFT INTERVAL
    L_interval_shifted = np.append(L_interval[1:], [max(pd) + 1], axis=0)
    L_bound = np.where(L_interval - L_interval_shifted < 0)[0][0] + 1
    # L_interval_restr = L_interval[:L_bound]
    # L_interval_shifted_restr = L_interval_shifted[:L_bound]
    # L_bound = np.argmax(L_interval_restr - L_interval_shifted_restr) + 1
    L_bound = min(L_bound, max_width)

    # RIGHT INTERVAL
    R_interval_shifted = np.append(R_interval[1:], [max(pd) + 1], axis=0)
    R_bound = np.where(R_interval - R_interval_shifted < 0)[0][0] + 1
    # R_interval_restr = R_interval[:R_bound]
    # R_interval_shifted_restr = R_interval_shifted[:R_bound]
    # R_bound = np.argmax(R_interval_restr - R_interval_shifted_restr) + 1
    R_bound = min(R_bound, max_width)

    return [max(MP - L_bound, 0), min(MP + R_bound, len(pd))]


def find_lower_v_domain(I, threshold_cut, max_height, min_persistence, it) -> Tuple[List, Optional[List]]:

    # For each slice (hosting a seed site):
    n, seed_site, HIoI = it

    # Standardized Local 1D pseudo-distribution for current HIoI:
    rows = slice(seed_site, min(seed_site + max_height, I.shape[0]))
    cols = slice(HIoI[0], HIoI[1])
    I_nb = I[rows, :].tocsc()[:, cols].toarray()

    Y = np.sum(I_nb, axis=1)
    Y /= max(Y)

    # Lower boundary:
    Y_hat = _compute_wQISA_predictions(Y, 5)  # Basically: average of a 2-"pixel" neighborhood

    # Peaks:
    if min_persistence is None:
        candida_bound = np.where(np.array(Y_hat) < threshold_cut)[0]
        if len(candida_bound) == 0:
            candida_bound = [len(Y_hat) - 1]

        # Vertical domain + no peak:
        return [seed_site, seed_site + candida_bound[0]], None

    _, _, loc_Maxima, loc_pers_of_Maxima = TDA.TDA(Y, min_persistence=min_persistence)
    candida_bound = [max(loc_Maxima)]

    # Consider as min_persistence is set to None:
    if len(loc_Maxima) < 2:
        candida_bound = np.where(np.array(Y_hat) < threshold_cut)[0]
        if len(candida_bound) == 0:
            candida_bound = [len(Y_hat) - 1]

    # Vertical domain + peaks:
    return [seed_site, seed_site + candida_bound[0]], list(np.array(loc_Maxima[:-1]) + seed_site)


def find_upper_v_domain(I, threshold_cut, max_height, min_persistence, it) -> Tuple[List, Optional[List]]:

    # For each slice (hosting a seed site):
    n, seed_site, HIoI = it

    # Standardized Local 1D pseudo-distribution for current HIoI:
    rows = slice(max(seed_site - max_height, 0), seed_site)
    cols = slice(HIoI[0], HIoI[1])
    I_nb = I[rows, :].tocsc()[:, cols].toarray()
    Y = np.flip(np.sum(I_nb, axis=1))
    Y /= max(Y)

    # Upper boundary:
    Y_hat = _compute_wQISA_predictions(Y, 5)

    # Peaks:
    if min_persistence is None:
        candida_bound = np.where(np.array(Y_hat) < threshold_cut)[0]
        if len(candida_bound) == 0:
            candida_bound = [len(Y_hat) - 1]

        # Vertical domain + no peak:
        return ([seed_site - candida_bound[0], seed_site], None)

    _, _, loc_Maxima, loc_pers_of_Maxima = TDA.TDA(Y, min_persistence=min_persistence)
    candida_bound = [max(loc_Maxima)]

    # Consider as min_persistence is set to None:
    if len(loc_Maxima) < 2:
        candida_bound = np.where(np.array(Y_hat) < threshold_cut)[0]
        if len(candida_bound) == 0:
            candida_bound = [len(Y_hat) - 1]

    # Vertical domain + peaks:
    return [seed_site - candida_bound[0], seed_site], list(seed_site - np.array(loc_Maxima[:-1]))


def find_HIoIs(pd, seed_sites, seed_site_bounds, max_width, map=map):
    """
    :param pd:                  acronym for pseudo-distribution, but can be any 1D array representing a uniformly-sample
                                scalar function works
    :param seed_sites:          maximum values in the pseudo-distribution (i.e., genomic coordinates hosting linear
                                patterns)
    :param seed_site_bounds:    for the i-th entry of seed_sites:
                                (*) seed_site_bounds[i] is the left boundary
                                (*) seed_site_bounds[i+1] is the right boundary
    :param max_width:           maximum width allowed
    :param map:                 alternative implementation of the built-in map function. Can be used to e.g. run this step in parallel by passing multiprocessing.Pool().map.
    :return:
    HIoIs                       list of lists, where each sublist is a pair consisting of the left and right boundaries
    """
    assert len(seed_site_bounds) == len(seed_sites) + 1

    iterable_input = [
        (seed_site, seed_site_bounds[num_MP], seed_site_bounds[num_MP + 1])
        for num_MP, seed_site in enumerate(seed_sites)
    ]

    HIoIs = list(map(partial(find_horizontal_domain, pd, max_width=max_width), iterable_input))

    # Handle possible overlapping intervals:
    for i in range(len(HIoIs) - 1):
        current_pair = HIoIs[i]
        next_pair = HIoIs[i + 1]

        if current_pair[1] > next_pair[0]:  # Check for intersection
            next_pair[0] = current_pair[1]  # Modify the second pair

    return HIoIs


def find_VIoIs(
    I,
    seed_sites,
    HIoIs,
    max_height,
    threshold_cut=0.1,
    min_persistence=0.20,
    where="lower",
    map=map,
):

    # Triplets to use in multiprocessing:
    iterable_input = [(n, seed_site, HIoI) for n, (seed_site, HIoI) in enumerate(zip(seed_sites, HIoIs))]

    # Lower-triangular part of the Hi-C matrix:
    if where == "lower":
        Vdomains_and_peaks = map(
            partial(find_lower_v_domain, I, threshold_cut, max_height, min_persistence),
            iterable_input,
        )

        VIoIs, peak_locs = list(zip(*Vdomains_and_peaks))

    # Upper-triangular part of the Hi-C matrix:
    elif where == "upper":
        # HIoIs = pool.map(partial(find_h_domain, pd), iterable_input)
        Vdomains_and_peaks = map(
            partial(find_upper_v_domain, I, threshold_cut, max_height, min_persistence),
            iterable_input,
        )

        VIoIs, peak_locs = list(zip(*Vdomains_and_peaks))

    return VIoIs, peak_locs
