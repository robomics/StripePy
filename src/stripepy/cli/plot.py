# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import logging
import pathlib
import random
from typing import Tuple, Union

import hictkpy
import numpy.typing as npt

import stripepy.plot


def _fetch_random_region(
    f: hictkpy.File,
    normalization: Union[str, None],
    seed: int,
    region_size: int = 2_500_000,
) -> Tuple[str, int, int, npt.NDArray]:
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
        start_pos = pos - offset
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


def run(
    contact_map: pathlib.Path,
    resolution: int,
    plot_type: str,
    output_name: pathlib.Path,
    region: Union[str, None],
    cmap: str,
    dpi: int,
    normalization: Union[str, None],
    force: bool,
    seed: int,
):
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

    if plot_type == "hic-matrix":
        fig, _ = stripepy.plot.hic_matrix(
            matrix,
            (start, end),
            title=f"{chrom}:{start}-{end}",
            cmap=cmap,
            log_scale=True,
        )
    else:
        raise NotImplementedError
    fig.savefig(output_name, dpi=dpi)
