# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import logging
import pathlib
import random
from typing import Dict, Optional, Tuple, Union

import hictkpy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def _parse_ucsc_region(region: str) -> Tuple[str, int, int]:
    try:
        chrom, _, pos = region.partition(":")
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


def _fetch_persistence_maximum_points(path: pathlib.Path, chrom: str, start: int, end: int) -> Dict[str, NDArray]:
    def fetch(v: NDArray[int], left_bound: int, right_bound: int) -> Tuple[NDArray[int], NDArray[int]]:
        assert left_bound >= 0
        assert right_bound >= left_bound

        idx = np.where((v >= left_bound) & (v <= right_bound))[0]
        return idx.astype(int), v[idx].astype(int)

    with ResultFile(path) as h5:
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

    if plot_type in {"pseudodistribution", "hic-matrix-with-sites"} and stripepy_hdf5 is None:
        raise RuntimeError('--stripepy-hdf5 is required when plot-type is "pseudodistribution"')

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
        chrom, start, end = _parse_ucsc_region(region)

    title = f"{chrom}:{start}-{end}"

    with contextlib.ExitStack() as ctx:
        if stripepy_hdf5 is not None:
            h5 = ctx.enter_context(ResultFile(stripepy_hdf5))
        else:
            h5 = None

        if plot_type == "hic-matrix":
            fig, _, _ = stripepy.plot.hic_matrix(
                matrix,
                (start, end),
                title=title,
                cmap=cmap,
                log_scale=True,
                with_colorbar=True,
            )
        elif plot_type == "pseudodistribution":
            data = _fetch_persistence_maximum_points(stripepy_hdf5, chrom, start, end)
            fig, _ = stripepy.plot.pseudodistribution(
                data["pseudodistribution_lt"],
                data["pseudodistribution_ut"],
                (start, end),
                f.resolution(),
                title=title,
                coords2scatter_lt=data["seeds_lt"],
                coords2scatter_ut=data["seeds_ut"],
            )
        elif plot_type == "hic-matrix-with-sites":
            data = _fetch_persistence_maximum_points(stripepy_hdf5, chrom, start, end)

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
                data["seeds_lt"] * f.resolution(),
                (start, end),
                location="lower",
                title=title,
                fig=fig,
                ax=axs[0],
            )

            stripepy.plot.plot_sites(
                data["seeds_ut"] * f.resolution(),
                (start, end),
                location="upper",
                title=title,
                fig=fig,
                ax=axs[1],
            )

            fig.tight_layout()

            fig.subplots_adjust(right=0.95)
            cbar_ax = fig.add_axes((0.95, 0.15, 0.015, 0.7))
            fig.colorbar(img, cax=cbar_ax)  # noqa
        elif plot_type == "hic-matrix-with-hioi":
            fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.4), sharey=True)

            geo_descriptors_lt = _fetch_geo_descriptors(h5, chrom, start, end, "LT")
            outlines_lt = [
                ((lb - start) // h5.resolution, (rb - start) // h5.resolution)
                for lb, rb in geo_descriptors_lt[["left_bound", "right_bound"]].itertuples(index=False)
            ]
            geo_descriptors_ut = _fetch_geo_descriptors(h5, chrom, start, end, "UT")
            outlines_ut = [
                ((lb - start) // h5.resolution, (rb - start) // h5.resolution)
                for lb, rb in geo_descriptors_ut[["left_bound", "right_bound"]].itertuples(index=False)
            ]

            _, _, img = stripepy.plot.hic_matrix(
                np.triu(matrix)
                + np.tril(stripepy.plot.mask_regions_1d(matrix, whitelist=outlines_lt, location="lower"), k=1),
                (start, end),
                title=title,
                cmap=cmap,
                log_scale=True,
                with_colorbar=False,
                fig=fig,
                ax=axs[0],
            )
            stripepy.plot.hic_matrix(
                np.tril(matrix)
                + np.triu(stripepy.plot.mask_regions_1d(matrix, whitelist=outlines_ut, location="upper"), k=1),
                (start, end),
                title=title,
                cmap=cmap,
                log_scale=True,
                with_colorbar=False,
                fig=fig,
                ax=axs[1],
            )

            rectangles = []
            for lb, ub in outlines_lt:
                x = start + (lb * h5.resolution)
                y = start + (lb * h5.resolution)
                width = (ub - lb + 1) * h5.resolution
                height = end - x
                rectangles.append((x, y, width, height))

            stripepy.plot.draw_boxes(rectangles, (start, end), color="blue", linestyle="dashed", fig=fig, ax=axs[0])

            rectangles = []
            for lb, ub in outlines_ut:
                x = start + (lb * h5.resolution)
                y = start + (lb * h5.resolution)
                width = (ub - lb + 1) * h5.resolution
                height = start - x
                rectangles.append((x, y, width, height))

            stripepy.plot.draw_boxes(rectangles, (start, end), color="blue", linestyle="dashed", fig=fig, ax=axs[1])

            axs[0].set(title="Lower Triangular")
            axs[1].set(title="Upper Triangular")

            fig.tight_layout()

            fig.subplots_adjust(right=0.95)
            cbar_ax = fig.add_axes((0.95, 0.15, 0.015, 0.7))
            fig.colorbar(img, cax=cbar_ax)

        else:
            raise NotImplementedError
    if plot_type not in {"hic-matrix-with-sites", "hic-matrix-with-hioi"}:
        fig.tight_layout()
    fig.savefig(output_name, dpi=dpi)
