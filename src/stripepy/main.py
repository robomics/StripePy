# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import multiprocessing as mp
import platform
import sys
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
    subcommand, kwargs, verbosity = setup.parse_args(sys.argv[1:] if args is None else args)

    log_file = kwargs.get("log_file")
    force = kwargs.get("force")
    matrix_file = kwargs.get("contact_map")
    with logging.ProcessSafeLogger(
        verbosity,
        log_file,
        force,
        matrix_file,
        print_welcome_message=subcommand != "view",
    ) as main_logger:
        try:
            _setup_matplotlib(subcommand, **kwargs)
            kwargs["main_logger"] = main_logger

            if subcommand == "call":
                return call.run(**kwargs, verbosity=verbosity)
            if subcommand == "download":
                return download.run(**kwargs)
            if subcommand == "plot":
                return plot.run(**kwargs)
            if subcommand == "view":
                return view.run(**kwargs)

            raise NotImplementedError

        except FileExistsError as e:
            import structlog

            structlog.get_logger().error(e)

            if args is not None:
                raise
            return 1

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
