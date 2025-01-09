# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT


def matplotlib_avail() -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    return True
