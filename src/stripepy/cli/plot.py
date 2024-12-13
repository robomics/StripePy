# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import logging
import pathlib
import random
from typing import Optional, Tuple, Union

import hictkpy
import numpy as np
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


def _find_seeds_in_RoI(
    seeds: NDArray[int], left_bound_RoI: int, right_bound_RoI: int
) -> Tuple[NDArray[int], NDArray[int]]:
    """
    Select seed coordinates that fall within the given left and right boundaries.

    Parameters
    ----------
    seeds: NDArray[int]
        a list with the seed coordinates
    left_bound_RoI: int
        left bound of the region of interest
    right_bound_RoI: int
        right bound of the region of interest

    Returns
    -------
    Tuple[NDArray[int], NDArray[int]]
        a tuple consisting of:

         * the indices of seed coordinates falling within the given boundaries
         * the coordinates of the selected seeds
    """

    assert left_bound_RoI >= 0
    assert right_bound_RoI >= left_bound_RoI

    # Find sites within the range of interest -- lower-triangular:
    ids_seeds_in_RoI = np.where((left_bound_RoI <= np.array(seeds)) & (np.array(seeds) <= right_bound_RoI))[0]
    seeds_in_RoI = np.array(seeds)[ids_seeds_in_RoI]

    return ids_seeds_in_RoI.astype(int), seeds_in_RoI.astype(int)


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

    if plot_type in {"pseudodistribution"} and stripepy_hdf5 is None:
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

    if plot_type == "hic-matrix":
        fig, _ = stripepy.plot.hic_matrix(
            matrix,
            (start, end),
            title=title,
            cmap=cmap,
            log_scale=True,
        )
    elif plot_type == "pseudodistribution":
        with ResultFile(stripepy_hdf5) as h5:
            pd_lt = h5.get(chrom, "pseudodistribution", "LT")["pseudodistribution"].to_numpy()
            pd_ut = h5.get(chrom, "pseudodistribution", "UT")["pseudodistribution"].to_numpy()

            min_persistence = h5.get_min_persistence(chrom)
            _, LT_ps_MPs = _find_seeds_in_RoI(
                np.sort(TDA(pd_lt, min_persistence=min_persistence)[2]), start // resolution, end // resolution
            )
            _, UT_ps_MPs = _find_seeds_in_RoI(
                np.sort(TDA(pd_ut, min_persistence=min_persistence)[2]), start // resolution, end // resolution
            )

        fig, _ = stripepy.plot.pseudodistribution(
            pd_lt,
            pd_ut,
            (start, end),
            f.resolution(),
            title=title,
            coords2scatter_lt=LT_ps_MPs,
            coords2scatter_ut=UT_ps_MPs,
        )
    else:
        raise NotImplementedError

    fig.tight_layout()
    fig.savefig(output_name, dpi=dpi)
