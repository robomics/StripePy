# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import functools


@functools.cache
def get_stderr():
    try:
        import rich.console

        return rich.console.Console(stderr=True)
    except ImportError:
        import sys

        return sys.stderr
