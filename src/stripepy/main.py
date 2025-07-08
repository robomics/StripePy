# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import importlib
import sys
from typing import List, Optional


def _setup_logger(subcommand: str, verbosity: str, kwargs):
    from stripepy.io import ProcessSafeLogger

    return ProcessSafeLogger(
        verbosity,
        path=kwargs.get("log_file"),
        force=kwargs.get("force"),
        matrix_file=kwargs.get("contact_map"),
        print_welcome_message=subcommand != "view",
        progress_bar_type=subcommand,
    )


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


def _dispatch_subcommand(subcommand: str, verbosity: str, **kwargs) -> int:
    """
    Call the appropriate entrypoint
    """

    from stripepy.cli import telemetry

    telem_span = kwargs.get("telem_span")

    try:
        entrypoint = importlib.import_module(f"stripepy.cli.{subcommand}")
    except Exception as e:  # noqa
        # This should never happen
        telemetry.set_exception(telem_span, e)
        raise NotImplementedError from e

    try:
        if subcommand == "call":
            ec = entrypoint.run(**kwargs, verbosity=verbosity)
        else:
            ec = entrypoint.run(**kwargs)

        telemetry.set_success(telem_span, ec)

        return ec

    except Exception as e:
        telemetry.set_exception(telem_span, e)
        raise


def main(args: Optional[List[str]] = None, no_telemetry: bool = False) -> int:
    # It is important that stripepy is not imported in the global namespace to enable coverage
    # collection when using multiprocessing
    from stripepy.cli import setup, telemetry

    # Parse CLI args
    try:
        subcommand, kwargs, verbosity = setup.parse_args(sys.argv[1:] if args is None else args)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1

    if subcommand == "help":
        return 0

    with contextlib.ExitStack() as ctx:
        main_logger = ctx.enter_context(_setup_logger(subcommand, verbosity, kwargs))
        telem_span = ctx.enter_context(telemetry.setup(subcommand, no_telemetry=no_telemetry))

        try:
            _setup_matplotlib(subcommand, **kwargs)
            kwargs["main_logger"] = main_logger
            kwargs["telem_span"] = telem_span
            return _dispatch_subcommand(subcommand, verbosity, **kwargs)

        except FileExistsError as e:
            import structlog

            # Do not print the full stack trace in case of FileExistsError
            # This make it easier to spot the names of the file(s) causing problems
            structlog.get_logger().error(e)
            if args is not None:
                raise
        except (RuntimeError, ImportError) as e:
            import structlog

            # Log the exception including its stack trace
            structlog.get_logger().exception(e)
            if args is not None:
                raise
        except Exception as e:  # noqa
            # Under normal operating conditions, StripePy should not raise exceptions other than
            # FileExistsError, RuntimeError, and ImportError.
            # Should that happen, log the exception with its stack trace and then re-raise it
            import structlog

            structlog.get_logger().exception(e)

            if args is not None:
                raise

        return 1


if __name__ == "__main__":
    sys.exit(main())
