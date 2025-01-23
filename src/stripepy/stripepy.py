# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import functools
import itertools
import pathlib
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as ss
import structlog
from numpy.typing import NDArray

from . import IO, plot
from .utils import common, finders, regressions, stripe
from .utils.common import pretty_format_elapsed_time, zero_columns, zero_rows
from .utils.multiprocess_sparse_matrix import SparseMatrix, get_shared_state
from .utils.persistence1d import PersistenceTable


def _log_transform(I: SparseMatrix) -> SparseMatrix:
    """
    Apply a log-transform to a sparse matrix ignoring (i.e. dropping) NaNs.

    Parameters
    ----------
    I : SparseMatrix
        the sparse matrix to be transformed

    Returns
    -------
    SparseMatrix
        the log-transformed sparse matrix
    """

    I.data[np.isnan(I.data)] = 0
    I.eliminate_zeros()
    Iproc = I.log1p()
    return Iproc


def _band_extraction(matrix: SparseMatrix, resolution: int, genomic_belt: int) -> ss.csr_matrix:
    """
    Given a symmetric sparse matrix in CSR format, do the following:

      * Zero (i.e. drop) all values that lie outside the first genomic_belt // resolution diagonals

    Parameters
    ----------
    matrix: SparseMatrix
        the upper-triangular sparse matrix to be processed
    resolution: int
        the genomic resolution of the sparse matrix I
    genomic_belt: int
        the width of the genomic belt to be extracted

    Returns
    -------
    ss.csr_matrix
    """

    assert resolution > 0
    assert genomic_belt > 0
    # assert ss.tril(matrix, k=-1).count_nonzero() == 0

    matrix_belt = genomic_belt // resolution
    if matrix_belt >= matrix.shape[0]:
        return matrix
    return ss.tril(matrix, k=matrix_belt, format="csr")


def _extract_RoIs(ut_matrix: ss.csr_matrix, RoI: Dict[str, List[int]]) -> Optional[ss.csr_matrix]:
    """
    Extract a region of interest (ROI) from the sparse matrix I

    Parameters
    ----------
    ut_matrix: ss.csr_matrix
        the upper-triangular sparse matrix to be processed
    RoI: Dict[str, List[int]]
        dictionary with the region of interest in matrix ('matrix') and genomic ('genomic') coordinates

    Returns
    -------
    Optional[ss.csr_matrix]
        sparse matrix with the same shape as the input matrix, but without interactions outside the given RoI
    """

    if RoI is None:
        return None

    assert isinstance(ut_matrix, ss.csr_matrix)

    i0, i1 = RoI["matrix"]
    idx = np.setdiff1d(np.arange(ut_matrix.shape[0]), np.arange(i0, i1 + 1))
    matrix_roi = zero_rows(ut_matrix, idx)
    matrix_roi = zero_columns(matrix_roi.tocsc(), idx)

    # make matrix symmetric
    return matrix_roi + ss.triu(matrix_roi, k=1, format="csc").T


def _compute_global_pseudodistribution(
    T: SparseMatrix, smooth: bool = True, decimal_places: int = 10
) -> NDArray[float]:
    """
    Given a sparse matrix T, marginalize it, scale the marginal so that maximum is 1, and then smooth it.

    Parameters
    ----------
    T: SparseMatrix
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

    pseudo_dist = np.squeeze(np.asarray(np.sum(T, axis=0)))  # marginalization
    pseudo_dist /= np.max(pseudo_dist)  # scaling
    if smooth:
        pseudo_dist = np.maximum(regressions._compute_wQISA_predictions(pseudo_dist, 11), pseudo_dist)  # smoothing

    if decimal_places >= 0:
        # We need to truncate FP numbers to ensure that later steps generate consistent results
        # even in the presence to very minor numeric differences on different platforms.
        return common.truncate_np(pseudo_dist, decimal_places)

    return pseudo_dist


def _check_neighborhood(
    values: NDArray[float],
    min_value: float = 0.1,
    neighborhood_size: int = 10,
    threshold_percentage: float = 0.85,
) -> NDArray[bool]:
    # TODO rea1991 Change neighborhood size from "matrix" to "genomic" (eg, default of 1 Mb)
    assert 0 <= min_value
    assert 0 < neighborhood_size
    assert 0 <= threshold_percentage <= 1

    if len(values) < neighborhood_size * 2:
        return np.full_like(values, False, dtype=bool)

    mask = np.full_like(values, True, dtype=bool)
    mask[:neighborhood_size] = False
    mask[-neighborhood_size:] = False

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
    location: str,
    logger=None,
) -> Tuple[pd.Series, pd.Series]:
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

    # If last maximum point was discarded, we need to remove the minimum point before it:
    if len(min_points_filtered) > 0 and len(min_points_filtered) == len(max_points_filtered):
        min_points_filtered = min_points_filtered.iloc[:-1]

    if logger is None:
        logger = structlog.get_logger()

    if len(max_points_filtered) == len(max_points):
        logger.bind(step=(2, 2, 3)).info("%s triangular part: no change in the number of seed sites", location)
    else:
        logger.bind(step=(2, 2, 3)).info(
            "%s triangular part: number of seed sites reduced from %d to %d",
            location,
            len(max_points),
            len(max_points_filtered),
        )

    return min_points_filtered, max_points_filtered


def _complement_persistent_minimum_points(
    pseudodistribution: NDArray[float],
    persistent_minimum_points: NDArray[int],
    persistent_maximum_points: NDArray[int],
) -> NDArray[int]:
    """
    TODO
    # Complement mPs with:
    # the global minimum (if any) that is to the left of the leftmost persistent maximum
    # AND
    # the global minimum (if any) that is to the right of the rightmost persistent maximum
    """
    assert len(persistent_maximum_points) > 0
    assert len(pseudodistribution) != 0

    i0 = persistent_maximum_points[0]
    i1 = persistent_maximum_points[-1]

    if i0 != 0:
        left_bound = np.argmin(pseudodistribution[:i0])
    else:
        left_bound = 0

    if i1 != len(pseudodistribution):
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


def step_1(
    matrix: ss.csr_matrix,
    genomic_belt: int,
    resolution: int,
    roi: Optional[Dict] = None,
    logger=None,
) -> Tuple[ss.csr_matrix, Optional[ss.csr_matrix], Optional[ss.csr_matrix]]:
    if logger is None:
        logger = structlog.get_logger().bind(step="IO")

    logger.bind(step=(1, 1)).info("focusing on a neighborhood of the main diagonal")
    matrix_proc = _band_extraction(matrix, resolution, genomic_belt)
    nnz0 = matrix.nnz
    nnz1 = matrix_proc.nnz
    delta = nnz0 - nnz1
    logger.bind(step=(1, 1)).info("removed %.2f%% of the non-zero entries (%d/%d)", (delta / nnz0) * 100, delta, nnz0)

    logger.bind(step=(1, 2)).info("applying log-transformation")

    # We need to extend the RoI to make sure we have all the data required to calculate the local pseudodistributions
    extension_window = genomic_belt // resolution
    if roi is None:
        extended_roi = None
    else:
        extended_roi = {
            "matrix": [
                max(0, roi["matrix"][0] - extension_window),
                min(matrix.shape[0], roi["matrix"][1] + extension_window),
            ]
        }

    roi_matrix_raw = _extract_RoIs(matrix_proc, extended_roi)
    matrix_proc = _log_transform(matrix_proc)

    logger.bind(step=(1, 3)).info("projecting interactions onto [1, 0]")
    scaling_factor = matrix_proc.max()
    matrix_proc /= scaling_factor

    if roi_matrix_raw is None:
        roi_matrix_proc = None
    else:
        roi_matrix_proc = _log_transform(roi_matrix_raw) / scaling_factor

    return matrix_proc, roi_matrix_raw, roi_matrix_proc  # noqa


def step_2(
    chrom_name: str,
    chrom_size: int,
    matrix: Optional[SparseMatrix],
    matrix_metadata: Optional[Dict],
    min_persistence: float,
    location: str,
    logger=None,
) -> Tuple[str, IO.Result]:
    assert location in {"lower", "upper"}

    if logger is None:
        logger = structlog.get_logger().bind(chrom=chrom_name, step=(2,))

    logger = logger.bind(location="LT" if location == "lower" else "UT")

    if matrix is None:
        assert matrix_metadata is not None
        matrix = get_shared_state(location, matrix_metadata).get()

    result = IO.Result(chrom_name, chrom_size)

    logger.bind(step=(2, 1, 0)).info("computing global 1D pseudo-distributions...")
    pseudodistribution = _compute_global_pseudodistribution(matrix, smooth=True)

    result.set_min_persistence(min_persistence)
    result.set("pseudodistribution", pseudodistribution, location)

    logger.bind(step=(2, 2, 0)).info(
        "detection of persistent maxima and corresponding minima for lower- and upper-triangular matrices..."
    )
    logger.bind(step=(2, 2, 0)).info("all maxima and their persistence")

    # All local minimum and maximum points:
    logger.bind(step=(2, 2, 1)).info(f"computing persistence for the {location} triangular part")
    persistence = PersistenceTable.calculate_persistence(pseudodistribution, min_persistence=0, sort_by="persistence")
    result.set("all_minimum_points", persistence.min.index.to_numpy(copy=True), location)
    result.set("all_maximum_points", persistence.max.index.to_numpy(copy=True), location)
    result.set("persistence_of_all_minimum_points", persistence.min.to_numpy(copy=True), location)
    result.set("persistence_of_all_maximum_points", persistence.max.to_numpy(copy=True), location)

    logger.bind(step=(2, 2, 2)).info(f"filtering low persistence values for the {location} triangular part")
    persistence.filter(min_persistence, method="greater")
    persistence.sort(by="position")

    logger.bind(step=(2, 2, 3)).info("removing seeds overlapping sparse regions")
    data = _filter_extrema_by_sparseness(
        matrix=matrix,
        min_points=persistence.min,
        max_points=persistence.max,
        location=location,
        logger=logger,
    )
    persistence = PersistenceTable(pers_of_min_points=data[0], pers_of_max_points=data[1], level_sets="upper")
    result.set("persistent_minimum_points", persistence.min.index.to_numpy(), location)
    result.set("persistent_maximum_points", persistence.max.index.to_numpy(), location)
    result.set("persistence_of_minimum_points", persistence.min.to_numpy(), location)
    result.set("persistence_of_maximum_points", persistence.max.to_numpy(), location)

    if len(persistence.max) == 0:
        return location, result

    logger.bind(step=(2, 3, 1)).info(f"{location}-triangular part: generating list of candidate stripes...")
    where = f"{location}_triangular"
    stripes = [stripe.Stripe(seed=x, top_pers=pers, where=where) for x, pers in persistence.max.items()]  # noqa
    logger.bind(step=(2, 3, 1)).info(f"{location}-triangular part: generated %d candidate stripes", len(stripes))
    result.set("stripes", stripes, location)

    return location, result


def step_3(
    result: IO.Result,
    matrix: Optional[SparseMatrix],
    matrix_metadata: Optional[Dict],
    resolution: int,
    genomic_belt: int,
    max_width: int,
    loc_pers_min: float,
    loc_trend_min: float,
    location: str,
    map_=map,
    logger=None,
) -> Tuple[str, IO.Result]:
    assert location in {"lower", "upper"}

    if logger is None:
        logger = structlog.get_logger().bind(chrom=result.chrom[0], step=(3,))

    logger = logger.bind(location="LT" if location == "lower" else "UT")

    if result.empty:
        logger.bind(step=(3,)).warning("no candidates found by step 2: returning immediately!")
        return location, result

    start_time = time.time()

    persistent_min_points = result.get("persistent_minimum_points", location)
    persistent_max_points = result.get("persistent_maximum_points", location)
    pseudodistribution = result.get("pseudodistribution", location)

    logger.bind(step=(3, 1)).info("width estimation")
    logger.bind(step=(3, 1, 1)).info("estimating candidate stripe widths")

    persistent_min_points_bounded = _complement_persistent_minimum_points(
        pseudodistribution, persistent_min_points, persistent_max_points
    )

    # DataFrame with the left and right boundaries for each seed site
    horizontal_domains = finders.find_HIoIs(
        pseudodistribution=pseudodistribution,
        seed_sites=persistent_max_points,
        seed_site_bounds=persistent_min_points_bounded,
        max_width=int(max_width / (2 * resolution)) + 1,
        logger=logger,
    )

    logger.bind(step=(3, 1, 2)).info("updating candidate stripes with width information")
    stripes = result.get("stripes", location)
    horizontal_domains.apply(
        lambda seed: stripes[seed.name].set_horizontal_bounds(seed["left_bound"], seed["right_bound"]),
        axis="columns",
    )

    domain_widths = horizontal_domains["right_bound"] - horizontal_domains["left_bound"]

    logger.bind(step=(3, 1)).info(
        "width estimation (mean=%.2f; std=%.2f) took %s",
        domain_widths.mean(),
        domain_widths.std(),
        common.pretty_format_elapsed_time(start_time),
    )

    logger.bind(step=(3, 2)).info("height estimation")
    start_time = time.time()

    logger.bind(step=(3, 2, 1)).info("estimating candidate stripe heights")
    vertical_domains = finders.find_VIoIs(
        matrix=matrix,
        matrix_metadata=matrix_metadata,
        seed_sites=persistent_max_points,
        horizontal_domains=horizontal_domains,
        max_height=int(genomic_belt / resolution),
        threshold_cut=loc_trend_min,
        min_persistence=loc_pers_min,
        location=location,
        map_=map_,
        logger=logger,
    )

    domain_heights = (vertical_domains["top_bound"] - vertical_domains["bottom_bound"]).abs()

    logger.bind(step=(3, 2, 2)).info("updating candidate stripes with height information")
    stripes = result.get("stripes", location)
    vertical_domains[["top_bound", "bottom_bound"]].apply(
        lambda seed: stripes[seed.name].set_vertical_bounds(seed["top_bound"], seed["bottom_bound"]),
        axis="columns",
    )

    logger.bind(step=(3, 2)).info(
        "height estimation (mean=%.2f; std=%.2f) took %s",
        domain_heights.mean(),
        domain_heights.std(),
        common.pretty_format_elapsed_time(start_time),
    )

    return location, result


def _step_4_helper(
    stripe: stripe.Stripe,
    matrix: Optional[SparseMatrix],
    matrix_metadata: Optional[Dict],
    window: int,
    location: str,
) -> stripe.Stripe:
    assert window >= 0

    if matrix is None:
        assert matrix_metadata is not None
        matrix = get_shared_state(location, matrix_metadata).get()

    stripe.compute_biodescriptors(matrix, window=window)

    return stripe


def step_4(
    stripes: List[stripe.Stripe],
    matrix: Optional[SparseMatrix],
    matrix_metadata: Optional[Dict],
    location: str,
    map_=map,
    logger=None,
    window: int = 3,
) -> Tuple[str, List[stripe.Stripe]]:
    if logger is None:
        logger = structlog.get_logger().bind(step=(4,))

    logger = logger.bind(location="LT" if location == "lower" else "UT")

    if len(stripes) == 0:
        logger.bind(step=(4,)).warning("no candidates found by step 2: returning immediately!")
        return location, stripes

    logger.bind(step=(4, 1)).info("computing stripe biological descriptors")

    return location, list(
        map_(
            functools.partial(
                _step_4_helper,
                matrix=matrix,
                matrix_metadata=matrix_metadata,
                window=window,
                location=location,
            ),
            stripes,
        )
    )


def _plot_matrix(
    chrom_name: str,
    chrom_size: int,
    resolution: int,
    start: int,
    end: int,
    matrix: NDArray,
    dest: pathlib.Path,
    matrix_type: str,
):
    t0 = time.time()
    plt = common._import_pyplot()

    if matrix_type == "raw":
        logger = structlog.get_logger().bind(chrom=chrom_name, step=(5, 1, 1))
    else:
        assert matrix_type == "processed"
        logger = structlog.get_logger().bind(chrom=chrom_name, step=(5, 1, 2))

    dummy_result = IO.Result(chrom_name, chrom_size)
    logger.info("plotting %s matrix", matrix_type)
    fig, _ = plot.plot(
        dummy_result,
        resolution=resolution,
        plot_type="matrix",
        start=start,
        end=end,
        matrix=matrix,
        log_scale=False,
    )

    fig.savefig(dest, dpi=256)
    plt.close(fig)
    logger.info("plotting %s matrix took %s", matrix_type, pretty_format_elapsed_time(t0))


def _plot_pseudodistribution(
    result: IO.Result,
    resolution: int,
    matrix: Optional[NDArray],
    output_folder: pathlib.Path,
):
    t0 = time.time()
    plt = common._import_pyplot()
    assert result.roi is not None

    logger = structlog.get_logger().bind(chrom=result.chrom[0], step=(5, 2, 1))
    logger.info("plotting pseudo-distributions")
    start, end = result.roi["genomic"]
    fig, _ = plot.plot(
        result,
        resolution,
        plot_type="pseudodistribution",
        start=start,
        end=end,
    )
    fig.savefig(output_folder / "pseudo_distribution.jpg", dpi=256)
    plt.close(fig)
    logger.info("plotting pseudo-distributions took %s", pretty_format_elapsed_time(t0))

    if matrix is None:
        return

    t0 = time.time()
    logger = logger.bind(step=(5, 2, 2))
    logger.info("plotting processed matrix with highlighted seed(s)")
    # Plot the region of interest of Iproc with over-imposed vertical lines for seeds:
    fig, _ = plot.plot(
        result,
        resolution,
        plot_type="matrix_with_seeds",
        matrix=matrix,
        start=start,
        end=end,
        log_scale=False,
    )
    fig.savefig(output_folder / "matrix_with_seeds.jpg", dpi=256)
    plt.close(fig)
    logger.info("plotting processed matrix with highlighted seed(s) took %s", pretty_format_elapsed_time(t0))


def _plot_hic_and_hois(
    result: IO.Result,
    resolution: int,
    matrix: Optional[NDArray],
    output_folder: pathlib.Path,
):
    assert result.roi is not None

    t0 = time.time()
    plt = common._import_pyplot()

    if matrix is None:
        return

    logger = structlog.get_logger().bind(chrom=result.chrom[0], step=(5, 3, 1))
    logger.info("plotting regions overlapping with candidate stripe(s)")
    start, end = result.roi["genomic"]
    fig, _ = plot.plot(
        result,
        resolution,
        "matrix_with_stripes_masked",
        start=start,
        end=end,
        matrix=matrix,
    )
    fig.savefig(output_folder / "all_domains.jpg", dpi=256)
    plt.close(fig)
    logger.info("plotting regions overlapping with candidate stripe(s) took %s", pretty_format_elapsed_time(t0))

    t0 = time.time()
    logger = logger.bind(step=(5, 3, 2))
    logger.info("plotting processed matrix with candidate stripe(s) highlighted")
    fig, _ = plot.plot(
        result,
        resolution,
        "matrix_with_stripes",
        start=start,
        end=end,
        matrix=matrix,
    )
    fig.savefig(output_folder / "all_candidates.jpg", dpi=256)
    plt.close(fig)

    logger.info(
        "plotting processed matrix with candidate stripe(s) highlighted took %s", pretty_format_elapsed_time(t0)
    )


def _plot_geo_descriptors(
    result: IO.Result,
    resolution: int,
    output_folder: pathlib.Path,
):
    t0 = time.time()
    plt = common._import_pyplot()

    logger = structlog.get_logger().bind(chrom=result.chrom[0], step=(5, 4, 1))
    logger.info("generating histograms for geo-descriptors")
    fig, _ = plot.plot(result, resolution, plot_type="geo_descriptors", start=0, end=result.chrom[1])
    fig.savefig(output_folder / "geo_descriptors.jpg", dpi=256)
    plt.close(fig)
    logger.info("generating histograms for geo-descriptors took %s", pretty_format_elapsed_time(t0))


def _marginalize_matrix_lt(
    matrix: SparseMatrix, seed: int, left_bound: int, right_bound: int, max_height: int
) -> NDArray:
    i1, i2 = seed, min(seed + max_height, matrix.shape[0])
    j1, j2 = left_bound, right_bound
    if isinstance(matrix, ss.csr_matrix):
        v = np.asarray(matrix[i1:i2, :].tocsc()[:, j1:j2].sum(axis=1)).flatten()
    else:
        v = np.asarray(matrix.tocsc()[:, j1:j2].tocsr()[i1:i2, :].sum(axis=1)).flatten()
    v /= v.max()

    return v


def _marginalize_matrix_ut(
    matrix: SparseMatrix, seed: int, left_bound: int, right_bound: int, max_height: int
) -> NDArray:
    i1, i2 = max(seed - max_height, 0), seed
    j1, j2 = left_bound, right_bound
    if isinstance(matrix, ss.csr_matrix):
        y = np.flip(np.asarray(matrix[i1:i2, :].tocsc()[:, j1:j2].sum(axis=1)).flatten())
    else:
        y = np.flip(np.asarray(matrix.tocsc()[:, j1:j2].tocsr()[i1:i2, :].sum(axis=1)).flatten())

    y /= y.max()

    return y


def _plot_local_pseudodistributions_helper(args):
    (
        i,
        seed,
        left_bound,
        right_bound,
        matrix,
        min_persistence,
        max_height,
        loc_trend_min,
        output_folder,
        location,
        chrom_name,
    ) = args

    if matrix is None:
        matrix = get_shared_state(location).get()

    plt = common._import_pyplot()
    structlog.get_logger().bind(chrom=chrom_name, step=(5, 5, i)).debug(
        "plotting local profile for seed %d (%s)", seed, location
    )

    if location == "LT":
        y = _marginalize_matrix_lt(matrix, seed, left_bound, right_bound, max_height)
    else:
        assert location == "UT"
        y = _marginalize_matrix_ut(matrix, seed, left_bound, right_bound, max_height)

    x = np.arange(seed, seed + len(y))
    y_hat = finders._compute_wQISA_predictions(y, 5)  # Basically: average of a 2-"pixel" neighborhood

    loc_maxima = PersistenceTable.calculate_persistence(
        y, min_persistence=min_persistence, sort_by="persistence"
    ).max.index.to_numpy()
    candidate_bound = [loc_maxima.max()]

    if len(loc_maxima) < 2:
        candidate_bound = np.where(y_hat < loc_trend_min)[0]
        if len(candidate_bound) == 0:
            candidate_bound = [len(y_hat) - 1]

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, color="red", linewidth=0.5, linestyle="solid")
    ax.plot(x, y_hat, color="black", linewidth=0.5, linestyle="solid")
    ax.plot(
        [seed + a for a in loc_maxima[:-1]],
        y[loc_maxima[:-1]],
        color="blue",
        marker=".",
        linestyle="",
        markersize=8 * 1.5,
    )
    ax.vlines(
        seed + candidate_bound[0],
        0.0,
        1.0,
        color="blue",
        linewidth=1.0,
        linestyle="dashed",
    )
    ax.set(xlim=(x[0], x[-1]), ylim=(0.0, 1.0))
    fig.tight_layout()
    fig.savefig(output_folder / f"{seed}_{location}_local_pseudodistribution.jpg", dpi=256)
    plt.close(fig)


def _plot_local_pseudodistributions(
    result: IO.Result,
    matrix_lt: Optional[SparseMatrix],
    matrix_ut: Optional[SparseMatrix],
    resolution: int,
    genomic_belt: int,
    loc_pers_min: float,
    loc_trend_min: float,
    output_folder: pathlib.Path,
    map_,
    logger,
):
    t0 = time.time()
    output_folder.mkdir()

    logger = logger.bind(step=(5, 5, 0))

    start, end = result.roi["genomic"]
    max_height = int(np.ceil(genomic_belt / resolution))

    df = result.get_stripe_geo_descriptors("LT")
    df = df[(df["left_bound"] * resolution >= start) & (df["right_bound"] * resolution <= end)]

    params = list(
        zip(
            itertools.count(1, 1),
            df["seed"],
            df["left_bound"],
            df["right_bound"],
            itertools.repeat(matrix_lt),
            itertools.repeat(loc_pers_min),
            itertools.repeat(max_height),
            itertools.repeat(loc_trend_min),
            itertools.repeat(output_folder),
            itertools.repeat("LT"),
            itertools.repeat(result.chrom[0]),
        )
    )

    if len(df) == 0:
        logger.bind(location="LT").info("no local profiles to plot!")
    else:
        logger.bind(location="LT").info("plotting local profiles for %d seed(s)", len(df))

    offset = len(df) + 1
    df = result.get_stripe_geo_descriptors("UT")
    df = df[(df["left_bound"] * resolution >= start) & (df["right_bound"] * resolution <= end)]

    params.extend(
        zip(
            itertools.count(offset, 1),
            df["seed"],
            df["left_bound"],
            df["right_bound"],
            itertools.repeat(matrix_ut),
            itertools.repeat(loc_pers_min),
            itertools.repeat(max_height),
            itertools.repeat(loc_trend_min),
            itertools.repeat(output_folder),
            itertools.repeat("UT"),
            itertools.repeat(result.chrom[0]),
        ),
    )

    if len(df) == 0:
        logger.bind(location="UT").info("no local profiles to plot!")
    else:
        logger.bind(location="UT").info("plotting local profiles for %d seed(s)", len(df))

    num_tasks = sum(1 for _ in map_(_plot_local_pseudodistributions_helper, params))
    if num_tasks > 0:
        logger.info("plotting %d profiles took %s", num_tasks, pretty_format_elapsed_time(t0))


def _plot_stripes_helper(args):
    t0 = time.time()
    plt = common._import_pyplot()
    i, matrix, result, resolution, start, end, cutoff, output_folder = args

    logger = structlog.get_logger().bind(chrom=result.chrom[0], step=(5, 6, i))
    logger.debug("plotting stripes with cutoff=%.2f", cutoff)

    fig, _ = plot.plot(
        result, resolution, "matrix_with_stripes", start=start, end=end, matrix=matrix, relative_change_threshold=cutoff
    )
    dest = output_folder / f"stripes_{cutoff:.2f}.jpg"
    fig.savefig(dest, dpi=256)
    plt.close(fig)
    logger.debug("plotting stripes with cutoff=%.2f took %s", cutoff, pretty_format_elapsed_time(t0))


def _get_stripes(result: IO.Result, resolution: int) -> pd.DataFrame:
    start, end = result.roi["genomic"]

    descriptors_lt = result.get_stripe_geo_descriptors("LT")
    descriptors_ut = result.get_stripe_geo_descriptors("UT")

    for col in ("seed", "left_bound", "right_bound", "top_bound", "bottom_bound"):
        descriptors_lt[col] *= resolution
        descriptors_ut[col] *= resolution

    descriptors_lt["rel_change"] = result.get_stripe_bio_descriptors("LT")["rel_change"].iloc[descriptors_lt.index]
    descriptors_ut["rel_change"] = result.get_stripe_bio_descriptors("UT")["rel_change"].iloc[descriptors_ut.index]

    df = pd.concat([descriptors_lt, descriptors_ut])
    return df[df["left_bound"].between(start, end, inclusive="both")]


def _plot_stripes(
    result: IO.Result,
    resolution: int,
    matrix: Optional[NDArray],
    output_folder: pathlib.Path,
    map_,
    logger,
):
    assert result.roi is not None

    t0 = time.time()

    if matrix is None:
        return

    logger = logger.bind(step=(5, 6, 0))

    df = _get_stripes(result, resolution)

    cutoffs = {}
    for cutoff in np.linspace(0, 15, 76):
        num_stripes = (df["rel_change"] >= cutoff).sum()
        if num_stripes not in cutoffs:
            cutoffs[num_stripes] = cutoff

    start, end = result.roi["genomic"]

    logger.info("plotting stripes using %d different cutoffs", len(cutoffs))
    list(
        map_(
            _plot_stripes_helper,
            zip(
                itertools.count(1, 1),
                itertools.repeat(matrix),
                itertools.repeat(result),
                itertools.repeat(resolution),
                itertools.repeat(start),
                itertools.repeat(end),
                cutoffs.values(),
                itertools.repeat(output_folder),
            ),
        ),
    )

    logger.info("plotting stripes using %d different cutoffs took %s", len(cutoffs), pretty_format_elapsed_time(t0))


def step_5(
    result: IO.Result,
    resolution: int,
    raw_matrix: SparseMatrix,
    proc_matrix: SparseMatrix,
    genomic_belt: int,
    loc_pers_min: float,
    loc_trend_min: float,
    output_folder: Optional[pathlib.Path],
    pool,
    logger=None,
):
    assert result.roi is not None

    if logger is None:
        logger = structlog.get_logger().bind(chrom=result.chrom[0], step=(5,))

    chrom_name, chrom_size = result.chrom
    start, end = result.roi["genomic"]

    for directory in ("1_preprocessing", "2_TDA", "3_shape_analysis", "4_biological_analysis"):
        (output_folder / chrom_name / directory).mkdir(parents=True, exist_ok=True)

    i0, i1 = result.roi["matrix"]
    raw_matrix_cropped = raw_matrix.tocsr()[i0:i1, :].tocsc()[:, i0:i1].toarray()
    proc_matrix_cropped = proc_matrix.tocsr()[i0:i1, :].tocsc()[:, i0:i1].toarray()

    tasks = []

    dest = output_folder / chrom_name / "1_preprocessing" / f"raw_matrix_{start}_{end}.jpg"
    tasks.append(pool.submit(_plot_matrix, *result.chrom, resolution, start, end, raw_matrix_cropped, dest, "raw"))

    if proc_matrix is not None:
        dest = output_folder / chrom_name / "1_preprocessing" / f"proc_matrix_{start}_{end}.jpg"
        tasks.append(
            pool.submit(_plot_matrix, *result.chrom, resolution, start, end, proc_matrix_cropped, dest, "processed")
        )

    tasks.append(
        pool.submit(
            _plot_pseudodistribution,
            result,
            resolution,
            proc_matrix_cropped,
            output_folder / chrom_name / "2_TDA",
        )
    )

    tasks.append(
        pool.submit(
            _plot_hic_and_hois,
            result,
            resolution,
            proc_matrix_cropped,
            output_folder / chrom_name / "3_shape_analysis",
        )
    )

    tasks.append(
        pool.submit(
            _plot_geo_descriptors,
            result,
            resolution,
            output_folder / chrom_name / "3_shape_analysis",
        )
    )

    _plot_local_pseudodistributions(
        result,
        ss.tril(proc_matrix, format="csc"),
        ss.triu(proc_matrix, format="csc"),
        resolution,
        genomic_belt,
        loc_pers_min,
        loc_trend_min,
        output_folder / chrom_name / "3_shape_analysis" / "local_pseudodistributions",
        pool.map,
        logger,
    )

    _plot_stripes(
        result,
        resolution,
        proc_matrix_cropped,
        output_folder / chrom_name / "4_biological_analysis",
        pool.map,
        logger,
    )

    for t in tasks:
        t.result()
