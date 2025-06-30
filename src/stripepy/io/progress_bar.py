# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

from typing import List, Tuple

import numpy as np
import pandas as pd


class _DummyProgressBar(object):
    """
    A progress bar class that does nothing.
    """

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def add_task(self, *args, **kwargs):  # noqa
        return None

    def update(self, *args, **kwargs):  # noqa
        pass

    def refresh(self):  # noqa
        pass


def _initialize_progress_bar_download():
    import rich.progress as rp

    from stripepy.io import get_stderr

    return rp.Progress(
        rp.TextColumn("[bold blue]{task.fields[name]}", justify="right"),
        rp.BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        rp.DownloadColumn(binary_units=True),
        "•",
        rp.TransferSpeedColumn(),
        "•",
        rp.TimeRemainingColumn(),
        console=get_stderr(),
        transient=True,
    )


def _initialize_progress_bar_call(longest_chrom_name: str, longest_step_name: str):
    import rich.progress as rp

    from stripepy.io import get_stderr

    chrom_field_fmt = f"task.fields[chrom]:>{len(longest_chrom_name)}"
    step_field_fmt = f"task.fields[step]:<{len(longest_step_name)}"

    return rp.Progress(
        rp.TextColumn("{task.fields[name]}", justify="right", style="bold blue"),
        rp.BarColumn(bar_width=None),
        rp.TextColumn("{task.percentage:>3.1f}%", style="progress.percentage"),
        rp.SpinnerColumn(),
        "•",
        rp.TextColumn(f"{{{chrom_field_fmt}}}:{{{step_field_fmt}}}", style="bold green"),
        "•",
        rp.TimeRemainingColumn(),
        console=get_stderr(),
        transient=True,
    )


def initialize_progress_bar(bar_type: str, **kwargs):
    """
    Attempt to initialize a progress bar using rich when appropriate.
    In case of failure, return a dummy progress bar that does nothing.
    """
    try:
        if bar_type == "call":
            return _initialize_progress_bar_call(**kwargs)
        if bar_type == "download":
            return _initialize_progress_bar_download()

        return _DummyProgressBar()

    except ImportError:
        return _DummyProgressBar()


def get_stripepy_call_progress_bar_weights(
    tasks: List[Tuple[str, int, bool]],
    include_plotting: bool,
    nproc: int,
):
    """
    Compute the weights used to update the progress bar.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
        index) chrom
        1) step_1
        2) step_2
        3) step_3
        4) step_4
        6) step_5
    """
    # These weights have been computed on a Linux machine (Ryzen 9 7950X3D) using 1 core to process
    # 4DNFI9GMP2J8.mcool at 10kbp

    # The following dictionaries use the value passed to --nproc as keys
    weights_without_plotting = {
        1: {
            "step_1": 40.06399703025818,
            "step_2": 3.18733811378479,
            "step_3": 48.011895179748535,
            "step_4": 11.020375490188599,
            "step_5": 0,
        },
        2: {
            "step_1": 14.151408195495605,
            "step_2": 1.775324821472168,
            "step_3": 25.758771181106567,
            "step_4": 6.573435306549072,
            "step_5": 0,
        },
        3: {
            "step_1": 21.056079149246216,
            "step_2": 1.8013880252838135,
            "step_3": 18.130164861679077,
            "step_4": 4.708052158355713,
            "step_5": 0,
        },
        4: {
            "step_1": 24.199871063232422,
            "step_2": 1.825760841369629,
            "step_3": 14.102145910263062,
            "step_4": 3.6864120960235596,
            "step_5": 0,
        },
        8: {
            "step_1": 32.215571880340576,
            "step_2": 1.8106591701507568,
            "step_3": 8.30887246131897,
            "step_4": 2.2084264755249023,
            "step_5": 0,
        },
        16: {
            "step_1": 33.82175922393799,
            "step_2": 1.8589463233947754,
            "step_3": 6.440566778182983,
            "step_4": 1.7117218971252441,
            "step_5": 0,
        },
    }

    weights_with_plotting = {
        1: {
            "step_1": 41.55667066574097,
            "step_2": 3.2346198558807373,
            "step_3": 47.96629214286804,
            "step_4": 11.205169677734375,
            "step_5": 142.8469741344452,
        },
        2: {
            "step_1": 7.261733770370483,
            "step_2": 1.7965450286865234,
            "step_3": 26.112728595733643,
            "step_4": 6.664078950881958,
            "step_5": 81.34452533721924,
        },
        3: {
            "step_1": 8.165515661239624,
            "step_2": 1.812662124633789,
            "step_3": 18.24624228477478,
            "step_4": 4.777962684631348,
            "step_5": 62.38240957260132,
        },
        4: {
            "step_1": 9.314571857452393,
            "step_2": 1.810298204421997,
            "step_3": 14.009633541107178,
            "step_4": 3.5988545417785645,
            "step_5": 49.95777225494385,
        },
        8: {
            "step_1": 14.522658824920654,
            "step_2": 1.82016921043396,
            "step_3": 8.170997142791748,
            "step_4": 2.1803231239318848,
            "step_5": 35.45014572143555,
        },
        16: {
            "step_1": 17.164432525634766,
            "step_2": 1.8949370384216309,
            "step_3": 5.90356183052063,
            "step_4": 1.733672857284546,
            "step_5": 34.46934366226196,
        },
    }

    if include_plotting:
        step_weights = weights_with_plotting
    else:
        step_weights = weights_without_plotting

    # Lookup the timings that best approximate the given number of processes
    avail_nproc = list(step_weights.keys())
    chosen_nproc = min(avail_nproc, key=lambda x: abs(x - nproc))
    step_weights = step_weights[chosen_nproc].copy()

    # Make weights relative
    # Weights for step 5 need special handling and are dealt with later on
    tot = sum(step_weights.values())
    plotting_time_pct = step_weights["step_5"] / tot
    tot -= step_weights.pop("step_5")
    step_weights = {k: v / tot for k, v in step_weights.items()}

    # Generate the list of weights for each chromosome, where baseline weights are adjusted
    # based on the chromosome size
    weights = []
    for _, size, skip in tasks:
        if not skip:
            weights.extend((size * w for w in step_weights.values()))
        else:
            weights.extend([0] * len(step_weights.values()))

    weights = np.array(weights).reshape(len(tasks), len(step_weights))
    df = pd.DataFrame(weights, columns=list(step_weights.keys()))

    # Initialize weights for step_5
    # Note that the runtime of step_5 is independent of chromosome size
    df["step_5"] = (df.sum().sum() * plotting_time_pct) / len(df)

    # Generate a mask to select rows corresponding to chromosomes that will be skipped
    # and set timings for those rows to 0
    mask = np.isclose(df.sum(axis="columns"), df["step_5"])
    df.loc[mask, "step_5"] = 0

    # Make weights relative
    df /= df.sum().sum()

    # Index dataframe based on chromosome name
    df["chrom"] = [chrom for chrom, _, _, in tasks]
    return df.set_index(["chrom"])
