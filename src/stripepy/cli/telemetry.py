# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import os
import platform
from typing import Dict, Optional, Tuple


def setup(
    subcommand: str,
    debug: bool = False,
    no_telemetry: bool = False,
):

    import structlog

    if no_telemetry:
        return contextlib.nullcontext()

    if "STRIPEPY_NO_TELEMETRY" in os.environ:
        no_telem_var = "STRIPEPY_NO_TELEMETRY"
    elif "NO_TELEMETRY" in os.environ:
        no_telem_var = "NO_TELEMETRY"
    else:
        no_telem_var = None

    if no_telem_var is not None:
        structlog.get_logger().debug(f"detected {no_telem_var} variable in environment: no telemetry will be collected")
        return contextlib.nullcontext()

    if not debug:
        import logging

        logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)

    structlog.get_logger().debug("setting up telemetry...")

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            Compression,
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        if debug:
            processor = BatchSpanProcessor(ConsoleSpanExporter())
        else:
            processor = BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint="https://stripepy-telemetry.paulsenlab.com:4319/v1/traces",
                    timeout=5,
                    compression=Compression.Gzip,
                )
            )

        provider = TracerProvider(resource=_generate_telemetry_resource())
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


def set_exception(span, exc: Exception):
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR))
        span.record_exception(exc, attributes={"exception.message": "", "exception.stacktrace": ""})
    except:  # noqa
        pass


def set_success(span, exit_code: int):
    try:
        from opentelemetry.trace import Status, StatusCode

        if exit_code == 0:
            status = Status(StatusCode.OK)
        else:
            status = Status(StatusCode.ERROR)

        span.set_status(status)
    except:  # noqa
        pass


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
