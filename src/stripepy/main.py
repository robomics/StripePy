# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import multiprocessing as mp
import platform
import sys
import traceback
from typing import List, Union

from .cli import call, download, logging, plot, setup, view


def _setup_matplotlib(subcommand: str, **kwargs):
    if subcommand not in {"call", "plot"}:
        return

    if subcommand == "call" and "roi" not in kwargs:
        return

    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        import structlog

        structlog.get_logger().warning("failed to configure matplotlib")
        return

    # This is very important, as some plotting operations are performed concurrently
    # using multiprocessing.
    # If the wrong backend is selected (e.g. tkinter) this can lead to the whole OS freezing
    matplotlib.use("Agg")
    plt.set_loglevel(level="warning")


def main(args: Union[List[str], None] = None):
    if args is None:
        args = sys.argv[1:]

    logging.setup_logger("INFO")
    try:
        subcommand, args, verbosity = setup.parse_args(args)
        _setup_matplotlib(subcommand, **args)

        if subcommand == "call":
            logging.setup_logger(
                verbosity,
                file=args.get("log_file"),
                force=args["force"],
                matrix_file=args["contact_map"],
            )
            return call.run(**args, verbosity=verbosity)

        logging.setup_logger(verbosity)
        if subcommand == "download":
            return download.run(**args)
        if subcommand == "plot":
            return plot.run(**args)
        if subcommand == "view":
            return view.run(**args)

        raise NotImplementedError

    except (RuntimeError, ImportError) as e:
        import structlog

        structlog.get_logger().exception(e)
        if args is not None:
            raise
        return 1


if __name__ == "__main__":
    if platform.system() == "Linux":
        mp.set_start_method("forkserver")
        mp.set_forkserver_preload(("numpy", "pandas", "scipy.sparse", "structlog"))
    else:
        mp.set_start_method("spawn")

    sys.exit(main())
