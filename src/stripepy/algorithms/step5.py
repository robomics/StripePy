# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import itertools
import pathlib
import time
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as ss
import structlog
from numpy.typing import NDArray

from stripepy import plot
from stripepy.algorithms.regressions import compute_wQISA_predictions
from stripepy.data_structures import (
    Persistence1DTable,
    Result,
    SparseMatrix,
    get_shared_state,
)
from stripepy.utils import import_pyplot, pretty_format_elapsed_time


def run(
    result: Result,
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
    """
    Plot the given matrix as a heatmap.
    """
    t0 = time.time()
    plt = import_pyplot()

    if matrix_type == "raw":
        logger = structlog.get_logger().bind(chrom=chrom_name, step=(5, 1, 1))
    else:
        assert matrix_type == "processed"
        logger = structlog.get_logger().bind(chrom=chrom_name, step=(5, 1, 2))

    dummy_result = Result(chrom_name, chrom_size)
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
    result: Result,
    resolution: int,
    matrix: Optional[NDArray],
    output_folder: pathlib.Path,
):
    """
    Plot the pseudo-distribution as a line plot.
    """
    t0 = time.time()
    plt = import_pyplot()
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
    result: Result,
    resolution: int,
    matrix: Optional[NDArray],
    output_folder: pathlib.Path,
):
    """
    Plot the given Hi-C matrix as a heatmap and highlight the candidate stripe domains.
    """
    assert result.roi is not None

    t0 = time.time()
    plt = import_pyplot()

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
    result: Result,
    resolution: int,
    output_folder: pathlib.Path,
):
    """
    Plot the histogram of the geometric descriptors (i.e. stripe widths and heights).
    """
    t0 = time.time()
    plt = import_pyplot()

    logger = structlog.get_logger().bind(chrom=result.chrom[0], step=(5, 4, 1))
    logger.info("generating histograms for geo-descriptors")
    fig, _ = plot.plot(result, resolution, plot_type="geo_descriptors", start=0, end=result.chrom[1])
    fig.savefig(output_folder / "geo_descriptors.jpg", dpi=256)
    plt.close(fig)
    logger.info("generating histograms for geo-descriptors took %s", pretty_format_elapsed_time(t0))


def _marginalize_matrix_lt(
    matrix: SparseMatrix,
    seed: int,
    left_bound: int,
    right_bound: int,
    max_height: int,
) -> NDArray:
    """
    Marginalize a lower-triangular matrix for the given region of interest..
    """
    i1, i2 = seed, min(seed + max_height, matrix.shape[0])
    j1, j2 = left_bound, right_bound
    if isinstance(matrix, ss.csr_matrix):
        v = np.asarray(matrix[i1:i2, :].tocsc()[:, j1:j2].sum(axis=1)).flatten()
    else:
        v = np.asarray(matrix.tocsc()[:, j1:j2].tocsr()[i1:i2, :].sum(axis=1)).flatten()
    v /= v.max()

    return v


def _marginalize_matrix_ut(
    matrix: SparseMatrix,
    seed: int,
    left_bound: int,
    right_bound: int,
    max_height: int,
) -> NDArray:
    """
    Marginalize a upper-triangular matrix for the given region of interest..
    """
    i1, i2 = max(seed - max_height, 0), seed
    j1, j2 = left_bound, right_bound
    if isinstance(matrix, ss.csr_matrix):
        y = np.flip(np.asarray(matrix[i1:i2, :].tocsc()[:, j1:j2].sum(axis=1)).flatten())
    else:
        y = np.flip(np.asarray(matrix.tocsc()[:, j1:j2].tocsr()[i1:i2, :].sum(axis=1)).flatten())

    y /= y.max()

    return y


def _plot_local_pseudodistributions_helper(args):
    """
    Worker function to plot the local pseudo-distribution as a line plot.
    """
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

    plt = import_pyplot()
    structlog.get_logger().bind(chrom=chrom_name, step=(5, 5, i)).debug(
        "plotting local profile for seed %d (%s)", seed, location
    )

    if location == "LT":
        y = _marginalize_matrix_lt(matrix, seed, left_bound, right_bound, max_height)
    else:
        assert location == "UT"
        y = _marginalize_matrix_ut(matrix, seed, left_bound, right_bound, max_height)

    x = np.arange(seed, seed + len(y))
    y_hat = compute_wQISA_predictions(y, 5)  # Basically: average of a 2-"pixel" neighborhood

    loc_maxima = Persistence1DTable.calculate_persistence(
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
    result: Result,
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
    """
    Plot the local pseudo-distributions for the stripes overlapping with the region of interest.
    Will produce one plot for each stripe.
    """
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
    plt = import_pyplot()
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


def _get_stripes(
    result: Result,
    resolution: int,
) -> pd.DataFrame:
    """
    Fetch stripes overlapping with the region of interest.
    """
    start, end = result.roi["genomic"]

    descriptors_lt = result.get_stripe_geo_descriptors("LT")
    descriptors_ut = result.get_stripe_geo_descriptors("UT")

    for col in ("seed", "left_bound", "right_bound", "top_bound", "bottom_bound"):
        descriptors_lt[col] *= resolution
        descriptors_ut[col] *= resolution

    descriptors_lt["rel_change"] = result.get_stripe_bio_descriptors("LT")["rel_change"].iloc[descriptors_lt.index]
    descriptors_ut["rel_change"] = result.get_stripe_bio_descriptors("UT")["rel_change"].iloc[descriptors_ut.index]

    df = pd.concat([descriptors_lt, descriptors_ut])

    left_bound_within_roi = df["left_bound"].between(start, end, inclusive="both")
    right_bound_within_roi = df["right_bound"].between(start, end, inclusive="both")
    return df[left_bound_within_roi & right_bound_within_roi]


def _plot_stripes(
    result: Result,
    resolution: int,
    matrix: Optional[NDArray],
    output_folder: pathlib.Path,
    map_,
    logger,
):
    """
    Plot the Hi-C map with stripes highlighted.
    Will produce one plot for each relative change cutoff between 0 and 15 with steps of 0.2.
    In case multiple steps would result in the same plot being produced, only the plot for the
    smallest cutoff is generated.
    """
    assert result.roi is not None

    t0 = time.time()

    if matrix is None:
        return

    logger = logger.bind(step=(5, 6, 0))

    df = _get_stripes(result, resolution)

    cutoffs = {}
    for cutoff in np.linspace(0, 15, 76):
        num_stripes = (df["rel_change"] >= cutoff).sum()  # noqa
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
