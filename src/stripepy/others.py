# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import os

import h5py
import hictkpy
import structlog


def _raise_invalid_bin_type_except(f: hictkpy.File):
    raise RuntimeError(f"Only files with a uniform bin size are supported, found \"{f.attributes()['bin-type']}\".")


def open_matrix_file_checked(path: os.PathLike, resolution: int) -> hictkpy.File:
    logger = structlog.get_logger()
    logger.info('validating file "%s" (%dbp)...', path, resolution)

    try:
        if not isinstance(resolution, int):
            raise TypeError("resolution must be an integer.")

        if resolution <= 0:
            raise ValueError("resolution must be greater than zero.")

        if hictkpy.is_scool_file(path):
            raise RuntimeError(".scool files are not currently supported.")
        if hictkpy.is_cooler(path):
            f = hictkpy.File(path)
            if f.resolution() == 0:
                _raise_invalid_bin_type_except(f)
            if f.resolution() != resolution:
                raise RuntimeError(f"expected {resolution} resolution, found {f.resolution()}.")
        else:
            f = hictkpy.MultiResFile(path)[resolution]
    except RuntimeError as e:
        raise RuntimeError(f'error opening file "{path}"') from e

    if f.attributes().get("bin-type", "fixed") != "fixed":
        _raise_invalid_bin_type_except(f)
    logger.info('file "%s" successfully validated', path)

    return f


def define_RoI(where_roi: str, chr_end: int, resolution: int):
    assert chr_end > 0
    assert resolution > 0

    # Region of Interest (RoI) in genomic and matrix coordinates:
    if where_roi == "middle":
        RoI_length = 2000000
        e1 = (chr_end - RoI_length) / 2
        e2 = e1 + RoI_length
        RoI = dict()
        RoI["genomic"] = [int(e1), int(e2), int(e1), int(e2)]  # genomic coordinates
        RoI["matrix"] = [int(roi / resolution) for roi in RoI["genomic"]]  # matrix coordinates
    elif where_roi == "start":
        RoI_length = 2000000
        e1 = 0
        e2 = e1 + RoI_length
        RoI = dict()
        RoI["genomic"] = [int(e1), int(e2), int(e1), int(e2)]  # genomic coordinates
        RoI["matrix"] = [int(roi / resolution) for roi in RoI["genomic"]]  # matrix coordinates
    else:
        RoI = None

    return RoI


# Define a function to visit groups and save terminal group names in a list
def save_terminal_groups(name, obj):
    group_names = []
    if isinstance(obj, h5py.Group):
        has_subgroups = any(isinstance(child_obj, h5py.Group) for _, child_obj in obj.items())
        if not has_subgroups:
            group_names.append(name)
    return group_names


# Define a function to visit datasets and save their names in a list
def save_all_datasets(name, obj):
    dataset_names = []
    if isinstance(obj, h5py.Dataset):  # check if obj is a group or dataset
        dataset_names.append(name)
    return dataset_names
