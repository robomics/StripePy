# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import importlib
import os
import platform
import sys
from typing import Dict, List, Optional, Tuple


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


def _try_fetch_dep_metadata(req: str) -> Optional[Tuple[str, str]]:
    try:
        from importlib.metadata import version

        from packaging.requirements import Requirement

        dep = Requirement(req)
        if dep.marker is not None and dep.marker != "all":
            return None

        if dep.name.startswith("opentelemetry"):
            return None

        return f"dependency.{dep.name}.version", version(dep.name)
    except:  # noqa
        return None


def _collect_deps() -> Dict[str, str]:
    from importlib.metadata import requires

    deps_metadata = {}

    for req in requires("stripepy-hic"):
        metadata = _try_fetch_dep_metadata(req)
        if metadata is not None:
            deps_metadata[metadata[0]] = metadata[1]
    return deps_metadata


def _generate_telemetry_resource():
    from opentelemetry.sdk.resources import (
        HOST_ARCH,
        PROCESS_RUNTIME_DESCRIPTION,
        PROCESS_RUNTIME_NAME,
        PROCESS_RUNTIME_VERSION,
        SERVICE_NAME,
        SERVICE_VERSION,
        OsResourceDetector,
        Resource,
    )

    from stripepy import __version__ as stripepy_version

    os_res = OsResourceDetector(raise_on_error=False).detect()

    res = Resource.create(
        {
            HOST_ARCH: platform.machine(),
            PROCESS_RUNTIME_NAME: platform.python_implementation(),
            PROCESS_RUNTIME_VERSION: platform.python_version(),
            PROCESS_RUNTIME_DESCRIPTION: platform.python_compiler(),
            SERVICE_NAME: "stripepy",
            SERVICE_VERSION: stripepy_version,
        }
    )

    deps = Resource.create(_collect_deps())

    return os_res.merge(deps).merge(res)


def _setup_telemetry(subcommand: str):
    import structlog

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        if "STRIPEPY_NO_TELEMETRY" in os.environ:
            no_telem_var = "STRIPEPY_NO_TELEMETRY"
        elif "NO_TELEMETRY" in os.environ:
            no_telem_var = "NO_TELEMETRY"
        else:
            no_telem_var = None

        if no_telem_var is not None:
            structlog.get_logger().debug(
                f"detected {no_telem_var} variable in environment: no telemetry will be collected"
            )
            return contextlib.nullcontext()

        structlog.get_logger().debug("setting up telemetry...")

        provider = TracerProvider(resource=_generate_telemetry_resource())
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        tracer = trace.get_tracer("stripepy")
        structlog.get_logger().debug("successfully configured StripePy for telemetry collection")

        return tracer.start_span(
            subcommand,
            record_exception=False,  # exceptions are recorded manually so that we can redact sensitive information
            set_status_on_exception=False,
        )

    except Exception as e:  # noqa
        structlog.get_logger().debug(f"failed to setup telemetry: no telemetry will be collected: {e}")
        return contextlib.nullcontext()


def _telemetry_set_exception(span, exc: Exception):
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(exc, attributes={"exception.message": "", "exception.stacktrace": ""})
    except:  # noqa
        pass


def _telemetry_set_success(span, exit_code: int):
    try:
        from opentelemetry.trace import Status, StatusCode

        if exit_code == 0:
            status = Status(StatusCode.OK)
        else:
            status = Status(StatusCode.ERROR)

        span.set_status(status)
    except:  # noqa
        pass


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

    telem_span = kwargs.get("telem_span")

    try:
        entrypoint = importlib.import_module(f"stripepy.cli.{subcommand}")
    except Exception as e:  # noqa
        # This should never happen
        _telemetry_set_exception(telem_span, e)
        raise NotImplementedError from e

    try:
        if subcommand == "call":
            ec = entrypoint.run(**kwargs, verbosity=verbosity)
        else:
            ec = entrypoint.run(**kwargs)

        _telemetry_set_success(telem_span, ec)

        return ec

    except Exception as e:
        _telemetry_set_exception(telem_span, e)
        raise


def main(args: Optional[List[str]] = None) -> int:
    # It is important that stripepy is not imported in the global namespace to enable coverage
    # collection when using multiprocessing
    from stripepy.cli import setup

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
        telem_span = ctx.enter_context(_setup_telemetry(subcommand))

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
        except (RuntimeError, ImportError) as e:
            import structlog

            # Log the exception including its stack trace
            structlog.get_logger().exception(e)
        except Exception as e:  # noqa
            # Under normal operating conditions, StripePy should not raise exceptions other than
            # FileExistsError, RuntimeError, and ImportError.
            # Should that happen, log the exception with its stack trace and then re-raise it
            import structlog

            structlog.get_logger().exception(e)

            raise

        if args is not None:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
