# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import math
import pathlib
import sys
from typing import Optional

import numpy as np
import pandas as pd

from stripepy.data_structures import ResultFile


def run(
    h5_file: pathlib.Path,
    relative_change_threshold: float,
    with_biodescriptors: bool,
    with_header: bool,
    coefficient_of_variation_threshold: Optional[float] = None,
    transform: Optional[str] = None,
    main_logger=None,
    telem_span=None,
) -> int:
    try:
        _configure_telemetry(
            telem_span,
            relative_change_threshold=relative_change_threshold,
            coefficient_of_variation_threshold=coefficient_of_variation_threshold,
        )
        skip_telemetry = telem_span is None
        with ResultFile(h5_file) as f:
            for chrom, size in f.chromosomes.items():
                _dump_stripes(
                    f,
                    chrom,
                    size,
                    f.resolution,
                    relative_change_threshold,
                    coefficient_of_variation_threshold,
                    transform,
                    with_biodescriptors,
                    with_header,
                    None if skip_telemetry else telem_span,
                )
                with_header = False
                skip_telemetry = True
    except BrokenPipeError as e:
        _handle_broken_pipe_error(e)

    return 0


def _handle_broken_pipe_error(e: BrokenPipeError):
    """
    Handle BrokenPipeError exceptions due to e.g. piping stripepy view into a command like head.
    This needs to deal with the following cases:
    - stdout is a tty: report the exception to the user
    - stdout is not a tty: forcefully close stdout and ignore BrokenPipeErrors.
      This is required to avoid errors like the following on macOS:
        Exception ignored on flushing sys.stdout:
        BrokenPipeError: [Errno 32] Broken pipe
        Command exited with non-zero status 120
    """

    if sys.stdout.isatty():
        raise e

    try:
        sys.stdout.close()
    except BrokenPipeError:
        # Make sure we exit with a clean code no matter what
        sys.exit(0)


def _configure_telemetry(
    span,
    relative_change_threshold: float,
    coefficient_of_variation_threshold: Optional[float],
):
    try:
        if not span.is_recording():
            return

        if coefficient_of_variation_threshold is None:
            coefficient_of_variation_threshold = math.nextafter(math.inf, 0)

        span.set_attribute("params.relative_change_threshold", relative_change_threshold)
        span.set_attribute("params.coefficient_of_variation_threshold", coefficient_of_variation_threshold)
    except:  # noqa
        pass


def _read_stripes(f: ResultFile, chrom: str) -> Optional[pd.DataFrame]:
    try:
        bio_lt = f.get(chrom, "bio_descriptors", "LT")
        bio_ut = f.get(chrom, "bio_descriptors", "UT")
        geo_lt = f.get(chrom, "geo_descriptors", "LT")
        geo_ut = f.get(chrom, "geo_descriptors", "UT")

        geo_lt["type"] = "lt"
        geo_ut["type"] = "ut"

        df1 = pd.concat([geo_lt, bio_lt], axis="columns")
        df2 = pd.concat([geo_ut, bio_ut], axis="columns")

        return pd.concat([df1, df2]).set_index("seed").sort_index(kind="stable")
    except Exception as e:
        missing_chrom_err_msg = "Unable to synchronously open object (component not found)"
        if f.format_version > 1 or missing_chrom_err_msg not in str(e):
            raise RuntimeError(f'failed to read stripes for chromosome "{chrom}": {e}')
        return None


def _stripes_to_bedpe(
    df: pd.DataFrame,
    chrom: str,
    size: int,
    resolution: int,
    transpose_policy: Optional[str],
    with_biodescriptors: bool,
) -> pd.DataFrame:
    num_stripes = len(df)

    start1_pos = np.minimum(df["left_bound"] * resolution, size)
    end1_pos = np.minimum(df["right_bound"] * resolution, size)
    start2_pos = np.minimum(df["top_bound"] * resolution, size)
    end2_pos = np.minimum(df["bottom_bound"] * resolution, size)

    if transpose_policy is not None:
        if transpose_policy == "transpose_to_ut":
            swap = df["type"] == "lt"
        elif transpose_policy == "transpose_to_lt":
            swap = df["type"] == "ut"
        else:
            raise NotImplementedError
    else:
        swap = None

    if swap is not None:
        start1_pos[swap], start2_pos[swap] = start2_pos[swap], start1_pos[swap]
        end1_pos[swap], end2_pos[swap] = end2_pos[swap], end1_pos[swap]

    dff = pd.DataFrame(
        {
            "chrom1": [chrom] * num_stripes,
            "start1": start1_pos,
            "end1": end1_pos,
            "chrom2": [chrom] * num_stripes,
            "start2": start2_pos,
            "end2": end2_pos,
        }
    )

    if with_biodescriptors:
        cols = ["left_bound", "right_bound", "top_bound", "bottom_bound", "type"]
        dff = pd.concat([dff, df.drop(columns=cols)], axis="columns")

    return dff


def _update_telemetry_attributes(span, h5_version: int, columns: pd.Index):
    try:
        if not span.is_recording():
            return

        span.set_attributes(
            {
                "params.columns": columns.tolist(),
                "params.result_file_format_version": h5_version,
            }
        )
    except:  # noqa
        pass


def _dump_stripes(
    f: ResultFile,
    chrom: str,
    size: int,
    resolution: int,
    relative_change_threshold: float,
    coefficient_of_variation_threshold: Optional[float],
    transpose_policy: str,
    with_biodescriptors: bool,
    with_header: bool,
    span_telem,
):
    df = _read_stripes(f, chrom)
    if df is None:
        return

    df = df[df["rel_change"] >= relative_change_threshold]
    if coefficient_of_variation_threshold is not None:
        df = df[df["cfx_of_variation"] < coefficient_of_variation_threshold]

    df = _stripes_to_bedpe(
        df,
        chrom,
        size,
        resolution,
        transpose_policy,
        with_biodescriptors,
    )

    _update_telemetry_attributes(
        span_telem,
        h5_version=f.format_version,
        columns=df.columns,
    )

    df.to_csv(
        sys.stdout,
        sep="\t",
        index=False,
        header=with_header,
        na_rep="nan",
    )
