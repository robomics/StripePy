# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import logging
import sys
from typing import List, Union

from .cli import call, download, plot, setup, view


def _setup_mpl_backend():
    # This is very important, as some plotting operations are performed concurrently
    # using multiprocessing.
    # If the wrong backend is selected (e.g. tkinter) this can lead to the whole OS freezing
    import matplotlib

    matplotlib.use("Agg")


def _setup_logger(level: str):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logging.getLogger().setLevel(level)


def main(args: Union[List[str], None] = None):
    if args is None:
        args = sys.argv[1:]

    subcommand, args = setup.parse_args(args)
    _setup_logger("INFO")  # TODO make tunable

    if subcommand == "call":
        _setup_mpl_backend()
        return call.run(**args)
    if subcommand == "download":
        return download.run(**args)
    if subcommand == "plot":
        _setup_mpl_backend()
        return plot.run(**args)
    if subcommand == "view":
        return view.run(**args)

    raise NotImplementedError


if __name__ == "__main__":
    main()
