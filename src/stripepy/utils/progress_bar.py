# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT


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

    from stripepy.IO import get_stderr

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

    from stripepy.IO import get_stderr

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
