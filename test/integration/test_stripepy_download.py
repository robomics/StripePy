# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import hashlib
import io
import json
import pathlib

import pytest

from stripepy import main


def _hash_file(path: pathlib.Path) -> str:
    hasher = hashlib.md5()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


@pytest.mark.end2end
class TestStripePyDownload:
    @staticmethod
    def test_list_only():
        args = ["download", "--list-only"]
        buff = io.StringIO()
        with contextlib.redirect_stdout(buff):
            main(args)

        data = json.loads(buff.getvalue())
        assert len(data) != 0

    @staticmethod
    def test_download_by_name(tmpdir):
        dest = pathlib.Path(tmpdir) / "out"

        args = ["download", "--name", "__results_v1", "--output", str(dest)]
        main(args)

        assert dest.is_file()

        assert _hash_file(dest) == "632b2a7a6e5c1a24dc3635710ed68a80"

    @staticmethod
    def test_download_random(tmpdir):
        dest = pathlib.Path(tmpdir) / "out"

        args = ["download", "--max-size", "2", "--output", str(dest)]
        main(args)

        assert dest.is_file()

        assert _hash_file(dest) == "d4ab74937dd9062efe4b2acc6ebc8780"
