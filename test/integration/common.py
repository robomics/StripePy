# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT


import json
import pathlib
from typing import Sequence

import pytest

from stripepy.IO import compare_result_files as _compare_result_files


def matplotlib_avail() -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    return True


def compare_result_files(reference: pathlib.Path, found: pathlib.Path, chroms: Sequence[str]):
    report = _compare_result_files(reference, found, chroms)
    if report["success"]:
        assert True
        return

    pytest.fail(json.dumps(report, indent=2))
