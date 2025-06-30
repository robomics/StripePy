# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import functools
import os
import pathlib
from typing import Any, Dict, Optional, Sequence

import hictkpy
import structlog
from pandas.testing import assert_frame_equal


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


def compare_result_files(
    reference: pathlib.Path,
    found: pathlib.Path,
    chroms: Optional[Sequence[str]] = None,
    raise_on_exception: bool = False,
) -> Dict[str, Any]:
    """
    Utility function to compare two result files.

    Parameters
    ----------
    reference: pathlib.Path
        path to the file to use as reference
    found: pathlib.Path
        path to the file to be compared with the reference
    chroms: Optional[Sequence[str]]
        one or more chromosomes to be compared.
        When not provided, all chromosomes will be compared.
    raise_on_exception: bool
        whether to raise an exception as soon as an error occurs.

    Returns
    -------
    Dict[str, Any]

    A dictionary with the outcome of the comparison.
    The files are identical if result["success"] is True.
    Otherwise, result["exception"] will contain the unhandled exception that caused the comparison
    to fail (if any), and result["chroms"] contains a dictionary with an entry for each chromosome
    compared. Each entry contains detailed information on the differences between the two files.
    """
    report = {"success": True, "exception": None, "chroms": {}}

    # we need to import this at the function scope to avoid issues with circular imports
    from stripepy.data_structures import ResultFile

    try:
        with ResultFile(reference) as f1, ResultFile(found) as f2:
            report["attributes"] = _compare_result_file_attributes(f1, f2, raise_on_exception)
            report["success"] = report["attributes"]["success"]
            if not report["success"]:
                return report

            if chroms is None:
                chroms = f1.chromosomes

            for chrom in chroms:
                report["chroms"][chrom] = {
                    "LT": _compare_result(f1, f2, chrom, "LT", raise_on_exception),
                    "UT": _compare_result(f1, f2, chrom, "UT", raise_on_exception),
                }
                if not report["chroms"][chrom]["LT"]["success"] or not report["chroms"][chrom]["UT"]["success"]:
                    report["success"] = False

    except Exception as e:
        if raise_on_exception:
            raise
        report["exception"] = str(e)
        report["success"] = False

    return report


def _compare_result_file_attributes(f1, f2, raise_on_exception: bool) -> Dict[str, Any]:
    result = {"success": True, "errors": []}
    try:
        for attr in ("assembly", "resolution", "format", "normalization", "chromosomes"):
            expected = getattr(f1, attr)
            found = getattr(f2, attr)

            if expected != found:
                result["errors"].append(
                    f'mismatched value for attribute "{attr}": expected "{expected}", found "{found}"'
                )
                result["success"] = False

        expected = f1.metadata
        found = f2.metadata

        expected.pop("min-chromosome-size")
        found.pop("min-chromosome-size")

        if expected != found:
            result["errors"].append(
                f'mismatched value for attribute "metadata": expected "{expected}", found "{found}"'
            )
            result["success"] = False

    except Exception as e:
        if raise_on_exception:
            raise
        result["success"] = False
        result["errors"].append(str(e))

    return result


def _compare_result_field(f1, f2, chrom: str, field: str, location: str, raise_on_exception: bool) -> Dict[str, Any]:
    assert chrom in f1.chromosomes
    df1 = f1.get(chrom, field, location)
    try:
        df2 = f2.get(chrom, field, location)
        assert_frame_equal(df1, df2, check_exact=False)
    except Exception as e:
        if raise_on_exception:
            raise
        return {"field": field, "success": False, "errors": [str(e)]}

    return {"field": field, "success": True, "errors": []}


def _compare_result(f1, f2, chrom: str, location: str, raise_on_exception: bool) -> Dict[str, Any]:
    fields = (
        "pseudodistribution",
        "all_minimum_points",
        "persistence_of_all_minimum_points",
        "all_maximum_points",
        "persistence_of_all_maximum_points",
        "stripes",
    )

    report = {"success": True}

    for field in fields:
        field_report = _compare_result_field(f1, f2, chrom, field, location, raise_on_exception)
        field_report.pop("field")
        report[field] = field_report

    for status in report.values():
        if not isinstance(status, dict):
            continue

        if not status["success"]:
            report["success"] = False  # noqa
            break

    return report
