# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import itertools
import logging
import pathlib
import random
import warnings
from typing import Dict, Optional, Tuple

import hictkpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import EngFormatter
from numpy.typing import NDArray

import stripepy.plot
from stripepy.IO import ResultFile
from stripepy.utils.TDA import TDA


def _generate_random_region(
    chroms: Dict[str, int], resolution: int, region_size: int = 2_500_000
) -> Tuple[str, int, int]:
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
            )

        offset = region_size // 2
        pos = random.randint(offset, chroms[chrom] - offset)
        start_pos = ((pos - offset) // resolution) * resolution
        end_pos = start_pos + region_size
        return (
            chrom,
            start_pos,
            end_pos,
        )

    raise RuntimeError(
        "Unable to randomly select a region to be plotted. Please manually select the desired region by passing the --region option."
    )


def _fetch_random_region(
    f: hictkpy.File,
    normalization: Optional[str],
    region_size: int = 2_500_000,
) -> Tuple[str, int, int, NDArray]:
    for attempt in range(10):
        chrom, start_pos, end_pos = _generate_random_region(
            f.chromosomes(include_ALL=False), f.resolution(), region_size
        )
        m = f.fetch(f"{chrom}:{start_pos}-{end_pos}", normalization=normalization).to_numpy()
        nnz = (np.isfinite(m) & (m != 0)).sum()

        if nnz / m.size >= 0.75:
            return chrom, start_pos, end_pos, m

    warnings.warn(
        "Failed to randomly select a genomic region with appropriate density for plotting.\n"
        "Continuing anyway.\n"
        "For best results, please manually provide the coordinates for a region to be plotted using parameter --region."
    )

    return chrom, start_pos, end_pos, m  # noqa


def _fetch_matrix(
    path: pathlib.Path, resolution: int, normalization: str, region: Optional[str]
) -> Tuple[str, int, int, NDArray]:
    f = hictkpy.MultiResFile(path)[resolution]

    if region is None:
        return _fetch_random_region(f, normalization)

    matrix = f.fetch(region, normalization=normalization).to_numpy()
    chrom, start, end = _parse_ucsc_region(region, f.chromosomes(include_ALL=False))

    return chrom, start, end, matrix


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

        idx = np.where((v >= left_bound) & (v < right_bound))[0]
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


def _plot_hic_matrix(
    contact_map: pathlib.Path,
    resolution: int,
    region: Optional[str],
    cmap: str,
    normalization: str,
    log_scale: bool,
    **kwargs,
) -> plt.Figure:
    chrom, start, end, matrix = _fetch_matrix(contact_map, resolution, normalization, region)

    fig, _, _ = stripepy.plot.hic_matrix(
        matrix,
        (start, end),
        cmap=cmap,
        log_scale=log_scale,
        with_colorbar=True,
    )

    fig.suptitle(f"{chrom}:{start}-{end}")
    fig.tight_layout()

    return fig


def _plot_hic_matrix_with_seeds(
    contact_map: pathlib.Path,
    stripepy_hdf5: pathlib.Path,
    resolution: int,
    region: Optional[str],
    cmap: str,
    normalization: str,
    log_scale: bool,
    **kwargs,
) -> plt.Figure:
    chrom, start, end, matrix = _fetch_matrix(contact_map, resolution, normalization, region)

    with ResultFile(stripepy_hdf5) as h5:
        data = _fetch_persistence_maximum_points(h5, chrom, start, end)
        resolution = h5.resolution

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.6), sharey=True)

    for ax in axs:
        _, _, img = stripepy.plot.hic_matrix(
            matrix,
            (start, end),
            cmap=cmap,
            log_scale=log_scale,
            with_colorbar=False,
            fig=fig,
            ax=ax,
        )

    stripepy.plot.plot_sites(
        data["seeds_lt"] * resolution,
        (start, end),
        location="lower",
        fig=fig,
        ax=axs[0],
    )

    stripepy.plot.plot_sites(
        data["seeds_ut"] * resolution,
        (start, end),
        location="upper",
        fig=fig,
        ax=axs[1],
    )

    fig.suptitle(f"{chrom}:{start}-{end}")
    fig.tight_layout()

    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes((0.95, 0.15, 0.015, 0.7))
    fig.colorbar(img, cax=cbar_ax)  # noqa

    return fig


def _plot_hic_matrix_with_stripes(
    contact_map: pathlib.Path,
    stripepy_hdf5: pathlib.Path,
    resolution: int,
    relative_change_threshold: Optional[float],
    region: Optional[str],
    cmap: str,
    normalization: str,
    override_height: Optional[int],
    mask_regions: bool,
    log_scale: bool,
    **kwargs,
) -> plt.Figure:
    chrom, start, end, matrix = _fetch_matrix(contact_map, resolution, normalization, region)

    with ResultFile(stripepy_hdf5) as h5:
        chrom_size = h5.chromosomes[chrom]
        geo_descriptors_lt = _fetch_geo_descriptors(h5, chrom, start, end, "LT")
        geo_descriptors_ut = _fetch_geo_descriptors(h5, chrom, start, end, "UT")
        if relative_change_threshold is not None:
            mask_lt = (
                h5.get(chrom, "bio_descriptors", "LT")["rel_change"].iloc[geo_descriptors_lt.index]
                >= relative_change_threshold
            )
            mask_ut = (
                h5.get(chrom, "bio_descriptors", "UT")["rel_change"].iloc[geo_descriptors_ut.index]
                >= relative_change_threshold
            )

            geo_descriptors_lt = geo_descriptors_lt[mask_lt]
            geo_descriptors_ut = geo_descriptors_ut[mask_ut]

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.6), sharey=True)

    outlines_lt = [
        (min(lb - start, chrom_size), min(rb - start, chrom_size), min(bb - tb, chrom_size))
        for lb, rb, bb, tb in geo_descriptors_lt[["left_bound", "right_bound", "bottom_bound", "top_bound"]].itertuples(
            index=False
        )
    ]

    outlines_ut = [
        (min(lb - start, chrom_size), min(rb - start, chrom_size), min(tb - bb, chrom_size))
        for lb, rb, bb, tb in geo_descriptors_ut[["left_bound", "right_bound", "bottom_bound", "top_bound"]].itertuples(
            index=False
        )
    ]

    if mask_regions:
        whitelist = [(x, y) for x, y, _ in outlines_lt]
        m1 = np.triu(matrix) + np.tril(
            stripepy.plot.mask_regions_1d(
                matrix,
                resolution,
                whitelist=whitelist,
                location="lower",
            ),
            k=1,
        )

        whitelist = [(x, y) for x, y, _ in outlines_ut]
        m2 = np.tril(matrix) + np.triu(
            stripepy.plot.mask_regions_1d(
                matrix,
                resolution,
                whitelist=whitelist,
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
        log_scale=log_scale,
        with_colorbar=False,
        fig=fig,
        ax=axs[0],
    )
    stripepy.plot.hic_matrix(
        m2,
        (start, end),
        cmap=cmap,
        log_scale=log_scale,
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
            height = min(end - x, override_height)
        rectangles.append((x, y, width, height))

    stripepy.plot.draw_boxes(rectangles, (start, end), color="blue", linestyle="dashed", fig=fig, ax=axs[0])

    rectangles = []
    for lb, ub, height in outlines_ut:
        x = min(start + lb, chrom_size)
        y = min(start + lb, chrom_size)
        width = min(ub - lb, chrom_size)
        if override_height is not None:
            height = min(start - x, override_height)
        rectangles.append((x, y, width, height))

    stripepy.plot.draw_boxes(rectangles, (start, end), color="blue", linestyle="dashed", fig=fig, ax=axs[1])

    axs[0].set(title="Lower Triangular")
    axs[1].set(title="Upper Triangular")

    fig.suptitle(f"{chrom}:{start}-{end}")
    fig.tight_layout()

    fig.subplots_adjust(right=0.94)
    cbar_ax = fig.add_axes((0.95, 0.15, 0.015, 0.7))
    fig.colorbar(img, cax=cbar_ax)

    return fig


def _plot_pseudodistribution(stripepy_hdf5: pathlib.Path, region: Optional[str], **kwargs) -> plt.Figure:
    with ResultFile(stripepy_hdf5) as h5:
        if region is None:
            chrom, start, end = _generate_random_region(h5.chromosomes, h5.resolution)
        else:
            chrom, start, end = _parse_ucsc_region(region, h5.chromosomes)
        data = _fetch_persistence_maximum_points(h5, chrom, start, end)
        resolution = h5.resolution

    fig, _ = stripepy.plot.pseudodistribution(
        data["pseudodistribution_lt"],
        data["pseudodistribution_ut"],
        region=(start, end),
        resolution=resolution,
        highlighted_points_lt=data["seeds_lt"] * resolution,
        highlighted_points_ut=data["seeds_ut"] * resolution,
    )

    fig.tight_layout()
    return fig


def _plot_stripe_dimension_distribution(
    stripepy_hdf5: pathlib.Path,
    region: Optional[str],
    **kwargs,
) -> plt.Figure:
    with ResultFile(stripepy_hdf5) as h5:
        if region is None:
            geo_descriptors_lt = _fetch_geo_descriptors_gw(h5, "LT")
            geo_descriptors_ut = _fetch_geo_descriptors_gw(h5, "UT")
        else:
            chrom, start, end = _parse_ucsc_region(region, h5.chromosomes)
            geo_descriptors_lt = _fetch_geo_descriptors(h5, chrom, start, end, "LT")
            geo_descriptors_ut = _fetch_geo_descriptors(h5, chrom, start, end, "UT")

        resolution = h5.resolution

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8), sharex="col", sharey="col")

    stripe_widths_lt = geo_descriptors_lt["right_bound"] - geo_descriptors_lt["left_bound"]
    stripe_heights_lt = geo_descriptors_lt["bottom_bound"] - geo_descriptors_lt["top_bound"]

    stripe_widths_ut = geo_descriptors_ut["right_bound"] - geo_descriptors_ut["left_bound"]
    stripe_heights_ut = geo_descriptors_ut["bottom_bound"] - geo_descriptors_ut["top_bound"]

    for ax in itertools.chain.from_iterable(axs):
        ax.xaxis.set_major_formatter(EngFormatter("b"))
        ax.xaxis.tick_bottom()

    axs[0][0].hist(stripe_widths_lt, bins=max(1, (stripe_widths_lt.max() - stripe_widths_lt.min()) // resolution))
    axs[0][1].hist(stripe_heights_lt, bins="auto")
    axs[1][0].hist(stripe_widths_ut, bins=max(1, (stripe_widths_ut.max() - stripe_widths_ut.min()) // resolution))
    axs[1][1].hist(stripe_heights_ut, bins="auto")

    axs[0][0].set(title="Stripe width distribution (lower triangle)", ylabel="Count")
    axs[0][1].set(title="Stripe height distribution (lower triangle)")
    axs[1][0].set(title="Stripe width distribution (upper triangle)", xlabel="Width (bp)", ylabel="Count")
    axs[1][1].set(title="Stripe height distribution (upper triangle)", xlabel="Height (bp)")

    if region is not None:
        fig.suptitle(f"{chrom}:{start}-{end}")

    fig.tight_layout()
    return fig


def run(plot_type: str, output_name: pathlib.Path, dpi: int, force: bool, **kwargs):
    if output_name.exists():
        if force:
            logging.debug('removing existing output file "%s"', output_name)
            output_name.unlink()
        else:
            raise RuntimeError(
                f'Refusing to overwrite file "{output_name}". Pass --force to overwrite existing file(s).'
            )

    if "seed" in kwargs:
        random.seed(kwargs["seed"])

    if plot_type in {"contact-map", "cm"}:
        plot_seeds = kwargs.pop("highlight_seeds")
        plot_stripes = kwargs.pop("highlight_stripes")
        ignore_stripe_heights = kwargs.pop("ignore_stripe_heights")

        if (plot_seeds or plot_stripes) and kwargs["stripepy_hdf5"] is None:
            raise RuntimeError("--stripepy-hdf5 is required when highlighting stripes or seeds.")

        if not plot_seeds and not plot_stripes:
            fig = _plot_hic_matrix(**kwargs)
        elif plot_seeds:
            fig = _plot_hic_matrix_with_seeds(**kwargs)
        else:
            if ignore_stripe_heights:
                kwargs["override_height"] = 2_500_000
                kwargs["mask_regions"] = True
            else:
                kwargs["override_height"] = None
                kwargs["mask_regions"] = False
            fig = _plot_hic_matrix_with_stripes(**kwargs)
    elif plot_type in {"pseudodistribution", "pd"}:
        fig = _plot_pseudodistribution(**kwargs)
    elif plot_type in {"stripe-hist", "hist"}:
        fig = _plot_stripe_dimension_distribution(**kwargs)
    else:
        raise NotImplementedError

    fig.savefig(output_name, dpi=dpi)
