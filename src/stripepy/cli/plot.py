# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import itertools
import logging
import pathlib
import random
from typing import Dict, Optional, Tuple, Union

import hictkpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import EngFormatter
from numpy.typing import NDArray

import stripepy.plot
from stripepy.IO import ResultFile
from stripepy.utils.TDA import TDA


def _fetch_random_region(
    f: hictkpy.File,
    normalization: Union[str, None],
    seed: int,
    region_size: int = 2_500_000,
) -> Tuple[str, int, int, NDArray]:
    # TODO deal with sparse regions
    random.seed(seed)
    chroms = f.chromosomes(include_ALL=False)
    chrom_names = list(chroms.keys())
    random.shuffle(chrom_names)

    for chrom in chrom_names:
        size = chroms[chrom]
        if size < region_size:
            continue

        if size == region_size:
            return (
                chrom,
                0,
                size,
                f.fetch(chrom, normalization=normalization).to_numpy(),
            )

        offset = region_size // 2
        pos = random.randint(offset, chroms[chrom] - offset)
        start_pos = ((pos - offset) // f.resolution()) * f.resolution()
        end_pos = start_pos + region_size
        return (
            chrom,
            start_pos,
            end_pos,
            f.fetch(f"{chrom}:{start_pos}-{end_pos}", normalization=normalization).to_numpy(),
        )

    raise RuntimeError(
        "Unable to randomly select a region to be plotted. Please manually select the desired region by passing the --region option."
    )


def _parse_ucsc_region(region: str, chromosomes: Dict[str, int]) -> Tuple[str, int, int]:
    try:
        chrom, sep, pos = region.partition(":")
        if len(sep) == 0 and len(pos) == 0:
            return chrom, 0, chromosomes[chrom]

        start_pos, _, end_pos = pos.partition("-")
        return chrom, int(start_pos.replace(",", "")), int(end_pos.replace(",", ""))
    except Exception as e:
        raise RuntimeError(f'Unable to parse region "{region}". Is the given region in UCSC format?') from e


def _validate_hdf5_result(hf: hictkpy.File, rf: ResultFile):
    if hf.resolution() != rf.resolution:
        raise RuntimeError(f'File "{hf.uri()}" and "{rf.path}" have different resolutions')

    if hf.chromosomes(include_ALL=False) != rf.chromosomes:
        raise RuntimeError(f'File "{hf.uri()}" and "{rf.path}" have different chromosomes')


def _fetch_geo_descriptors(
    h5: ResultFile,
    chrom: str,
    left_bound: int,
    right_bound: int,
    location: str,
) -> pd.DataFrame:
    assert location in {"LT", "UT"}
    assert left_bound >= 0
    assert right_bound >= left_bound

    df = h5.get(chrom, "geo_descriptors", location)

    for col in df.columns:
        if col == "top_persistence":
            continue
        df[col] = np.minimum(df[col] * h5.resolution, h5.chromosomes[chrom])

    return df[df["seed"].between(left_bound, right_bound, inclusive="both")]


def _fetch_geo_descriptors_gw(
    h5: ResultFile,
    location: str,
) -> pd.DataFrame:
    assert location in {"LT", "UT"}

    dfs = []
    for chrom in h5.chromosomes:
        try:
            df = h5.get(chrom, "geo_descriptors", location)
            for col in df.columns:
                if col == "top_persistence":
                    continue
                df[col] = np.minimum(df[col] * h5.resolution, h5.chromosomes[chrom])
            dfs.append(df)

        except KeyError:
            pass

    return pd.concat(dfs).reset_index(drop=True)


def _fetch_persistence_maximum_points(h5: ResultFile, chrom: str, start: int, end: int) -> Dict[str, NDArray]:
    def fetch(v: NDArray[int], left_bound: int, right_bound: int) -> Tuple[NDArray[int], NDArray[int]]:
        assert left_bound >= 0
        assert right_bound >= left_bound

        idx = np.where((v >= left_bound) & (v <= right_bound))[0]
        return idx.astype(int), v[idx].astype(int)

    pd_lt = h5.get(chrom, "pseudodistribution", "LT")["pseudodistribution"].to_numpy()
    pd_ut = h5.get(chrom, "pseudodistribution", "UT")["pseudodistribution"].to_numpy()

    min_persistence = h5.get_min_persistence(chrom)
    lt_idx, lt_seeds = fetch(
        np.sort(TDA(pd_lt, min_persistence=min_persistence)[2]), start // h5.resolution, end // h5.resolution
    )
    ut_idx, ut_seeds = fetch(
        np.sort(TDA(pd_ut, min_persistence=min_persistence)[2]), start // h5.resolution, end // h5.resolution
    )

    return {
        "pseudodistribution_lt": pd_lt,
        "pseudodistribution_ut": pd_ut,
        "seeds_lt": lt_seeds,
        "seeds_ut": ut_seeds,
        "seed_indices_lt": lt_idx,
        "seed_indices_ut": ut_idx,
    }


def _plot_hic_matrix(matrix: NDArray, chrom: str, start: int, end: int, cmap: str) -> plt.Figure:
    fig, _, _ = stripepy.plot.hic_matrix(
        matrix,
        (start, end),
        title=f"{chrom}:{start}-{end}",
        cmap=cmap,
        log_scale=True,
        with_colorbar=True,
    )

    return fig


def _plot_pseudodistribution(h5: ResultFile, chrom: str, start: int, end: int) -> plt.Figure:
    data = _fetch_persistence_maximum_points(h5, chrom, start, end)
    fig, _ = stripepy.plot.pseudodistribution(
        data["pseudodistribution_lt"],
        data["pseudodistribution_ut"],
        (start, end),
        h5.resolution,
        title=f"{chrom}:{start}-{end}",
        coords2scatter_lt=data["seeds_lt"],
        coords2scatter_ut=data["seeds_ut"],
    )

    fig.tight_layout()
    return fig


def _plot_hic_matrix_with_sites(
    h5: ResultFile, matrix: NDArray, chrom: str, start: int, end: int, cmap: str
) -> plt.Figure:
    data = _fetch_persistence_maximum_points(h5, chrom, start, end)

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.4), sharey=True)

    for ax in axs:
        _, _, img = stripepy.plot.hic_matrix(
            matrix,
            (start, end),
            cmap=cmap,
            log_scale=True,
            with_colorbar=False,
            fig=fig,
            ax=ax,
        )

    stripepy.plot.plot_sites(
        data["seeds_lt"] * h5.resolution,
        (start, end),
        location="lower",
        fig=fig,
        ax=axs[0],
    )

    stripepy.plot.plot_sites(
        data["seeds_ut"] * h5.resolution,
        (start, end),
        location="upper",
        fig=fig,
        ax=axs[1],
    )

    fig.suptitle(f"{chrom}:{start}-{end}")
    fig.tight_layout()

    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes((0.95, 0.15, 0.015, 0.7))
    fig.colorbar(img, cax=cbar_ax)  # noqa

    return fig


def _plot_hic_matrix_with_stripes(
    h5: ResultFile,
    matrix: NDArray,
    chrom: str,
    start: int,
    end: int,
    cmap: str,
    override_height: Optional[int] = None,
    mask_regions: bool = False,
) -> plt.Figure:
    fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.4), sharey=True)

    chrom_size = h5.chromosomes[chrom]

    geo_descriptors_lt = _fetch_geo_descriptors(h5, chrom, start, end, "LT")
    outlines_lt = [
        (min(lb - start, chrom_size), min(rb - start, chrom_size), min(bb - tb, chrom_size))
        for lb, rb, bb, tb in geo_descriptors_lt[["left_bound", "right_bound", "bottom_bound", "top_bound"]].itertuples(
            index=False
        )
    ]
    geo_descriptors_ut = _fetch_geo_descriptors(h5, chrom, start, end, "UT")
    outlines_ut = [
        (min(lb - start, chrom_size), min(rb - start, chrom_size), min(tb - bb, chrom_size))
        for lb, rb, bb, tb in geo_descriptors_ut[["left_bound", "right_bound", "bottom_bound", "top_bound"]].itertuples(
            index=False
        )
    ]

    if mask_regions:
        m1 = np.triu(matrix) + np.tril(
            stripepy.plot.mask_regions_1d(
                matrix,
                whitelist=[
                    (
                        x,
                        y,
                    )
                    for x, y, _ in outlines_lt
                ],
                location="lower",
            ),
            k=1,
        )
        m2 = np.tril(matrix) + np.triu(
            stripepy.plot.mask_regions_1d(
                matrix,
                whitelist=[
                    (
                        x,
                        y,
                    )
                    for x, y, _ in outlines_ut
                ],
                location="upper",
            ),
            k=1,
        )
    else:
        m1 = matrix
        m2 = matrix

    _, _, img = stripepy.plot.hic_matrix(
        m1,
        (start, end),
        cmap=cmap,
        log_scale=True,
        with_colorbar=False,
        fig=fig,
        ax=axs[0],
    )
    stripepy.plot.hic_matrix(
        m2,
        (start, end),
        cmap=cmap,
        log_scale=True,
        with_colorbar=False,
        fig=fig,
        ax=axs[1],
    )

    rectangles = []
    for lb, ub, height in outlines_lt:
        x = min(start + lb, chrom_size)
        y = min(start + lb, chrom_size)
        width = min(ub - lb, h5.chromosomes[chrom])
        if override_height is not None:
            height = min(end - start, override_height)
        rectangles.append((x, y, width, height))

    stripepy.plot.draw_boxes(rectangles, (start, end), color="blue", linestyle="dashed", fig=fig, ax=axs[0])

    rectangles = []
    for lb, ub, height in outlines_ut:
        x = min(start + lb, chrom_size)
        y = min(start + lb, chrom_size)
        width = min(ub - lb, chrom_size)
        if override_height is not None:
            height = min(end - start, override_height)
        rectangles.append((x, y, width, height))

    stripepy.plot.draw_boxes(rectangles, (start, end), color="blue", linestyle="dashed", fig=fig, ax=axs[1])

    axs[0].set(title="Lower Triangular")
    axs[1].set(title="Upper Triangular")

    fig.suptitle(f"{chrom}:{start}-{end}")
    fig.tight_layout()

    fig.subplots_adjust(right=0.95)
    cbar_ax = fig.add_axes((0.95, 0.15, 0.015, 0.7))
    fig.colorbar(img, cax=cbar_ax)

    return fig


def _plot_stripe_dimension_distribution(
    h5: ResultFile, chrom: Optional[str], start: Optional[int], end: Optional[int]
) -> plt.Figure:
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8), sharex="col", sharey="col")

    if chrom is None:
        assert start is None
        assert end is None
        geo_descriptors_lt = _fetch_geo_descriptors_gw(h5, "LT")
        geo_descriptors_ut = _fetch_geo_descriptors_gw(h5, "UT")
    else:
        assert start is not None
        assert end is not None
        geo_descriptors_lt = _fetch_geo_descriptors(h5, chrom, start, end, "LT")
        geo_descriptors_ut = _fetch_geo_descriptors(h5, chrom, start, end, "UT")

    stripe_widths_lt = geo_descriptors_lt["right_bound"] - geo_descriptors_lt["left_bound"]
    stripe_heights_lt = geo_descriptors_lt["bottom_bound"] - geo_descriptors_lt["top_bound"]

    stripe_widths_ut = geo_descriptors_ut["right_bound"] - geo_descriptors_ut["left_bound"]
    stripe_heights_ut = geo_descriptors_ut["bottom_bound"] - geo_descriptors_ut["top_bound"]

    for ax in itertools.chain.from_iterable(axs):
        ax.xaxis.set_major_formatter(EngFormatter("b"))
        ax.xaxis.tick_bottom()

    axs[0][0].hist(stripe_widths_lt, bins=max(1, (stripe_widths_lt.max() - stripe_widths_lt.min()) // h5.resolution))
    axs[0][1].hist(stripe_heights_lt, bins="auto")
    axs[1][0].hist(stripe_widths_ut, bins=max(1, (stripe_widths_ut.max() - stripe_widths_ut.min()) // h5.resolution))
    axs[1][1].hist(stripe_heights_ut, bins="auto")

    axs[0][0].set(title="Stripe width distribution (lower triangle)", ylabel="Count")
    axs[0][1].set(title="Stripe height distribution (lower triangle)")
    axs[1][0].set(title="Stripe width distribution (upper triangle)", xlabel="Width (bp)", ylabel="Count")
    axs[1][1].set(title="Stripe height distribution (upper triangle)", xlabel="Height (bp)")

    if chrom is not None:
        assert start is not None
        assert end is not None
        fig.suptitle(f"{chrom}:{start}-{end}")

    fig.tight_layout()
    return fig


def run(
    contact_map: pathlib.Path,
    resolution: int,
    plot_type: str,
    output_name: pathlib.Path,
    stripepy_hdf5: Optional[pathlib.Path],
    region: Optional[str],
    cmap: str,
    dpi: int,
    normalization: Optional[str],
    force: bool,
    seed: int,
):

    plot_types_requiring_hdf5_file = {
        "pseudodistribution",
        "hic-matrix-with-sites",
        "hic-matrix-with-stipes",
        "stripe-dimension-distributions",
    }

    if stripepy_hdf5 is None and plot_type in plot_types_requiring_hdf5_file:
        raise RuntimeError(
            f"--stripepy-hdf5 is required when plot-type is one of {', '.join(plot_types_requiring_hdf5_file)}"
        )

    if output_name.exists():
        if force:
            logging.debug('removing existing output file "%s"', output_name)
            output_name.unlink()
        else:
            raise RuntimeError(
                f'Refusing to overwrite file "{output_name}". Pass --force to overwrite existing file(s).'
            )

    f = hictkpy.MultiResFile(contact_map)[resolution]

    if region is None:
        chrom, start, end, matrix = _fetch_random_region(f, normalization, seed)
    else:
        matrix = f.fetch(region, normalization=normalization).to_numpy()
        chrom, start, end = _parse_ucsc_region(region, f.chromosomes(include_ALL=False))

    with contextlib.ExitStack() as ctx:
        if stripepy_hdf5 is not None:
            h5 = ctx.enter_context(ResultFile(stripepy_hdf5))
        else:
            h5 = None

        if plot_type == "hic-matrix":
            assert matrix is not None
            fig = _plot_hic_matrix(matrix, chrom, start, end, cmap)
        elif plot_type == "pseudodistribution":
            assert h5 is not None
            fig = _plot_pseudodistribution(h5, chrom, start, end)
        elif plot_type == "hic-matrix-with-sites":
            assert matrix is not None
            assert h5 is not None
            fig = _plot_hic_matrix_with_sites(h5, matrix, chrom, start, end, cmap)
        elif plot_type == "hic-matrix-with-hioi":
            assert matrix is not None
            assert h5 is not None
            fig = _plot_hic_matrix_with_stripes(
                h5, matrix, chrom, start, end, cmap, override_height=2_000_000, mask_regions=True
            )
        elif plot_type == "hic-matrix-with-stripes":
            assert matrix is not None
            assert h5 is not None
            fig = _plot_hic_matrix_with_stripes(h5, matrix, chrom, start, end, cmap)
        elif plot_type == "stripe-dimension-distributions":
            assert h5 is not None
            if region is None:
                chrom = None
                start = None
                end = None
            fig = _plot_stripe_dimension_distribution(h5, chrom, start, end)
        else:
            raise NotImplementedError

    fig.savefig(output_name, dpi=dpi)
