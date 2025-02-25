# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import functools
import os

import hictkpy
import structlog


@functools.cache
def get_stderr():
    try:
        import rich.console

        return rich.console.Console(stderr=True)
    except ImportError:
        import sys

        return sys.stderr


def _raise_invalid_bin_type_except(f: hictkpy.File):
    raise RuntimeError(f"Only files with a uniform bin size are supported, found \"{f.attributes()['bin-type']}\".")


def open_matrix_file_checked(path: os.PathLike, resolution: int, logger=None) -> hictkpy.File:
    """
    Open a file in one of the formats supported by hictkpy and check that it satisfies StripePy requirements.

    Parameters
    ----------
    path: os.PathLike
        path to the matrix file
    resolution: int
        resolution to be used to open the matrix file
    logger:
        logger

    Returns
    -------
    hictkpy.File
        the given file opened with hictkpy
    """
    if logger is None:
        logger = structlog.get_logger()

    logger.bind(step="IO")
    logger.info('validating file "%s" (%dbp)', path, resolution)

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
