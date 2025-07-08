# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

from .common import stripepy_main


@pytest.mark.end2end
class TestStripePyCLI:
    @staticmethod
    def test_stripepy_license():
        f = io.StringIO()
        with redirect_stdout(f):
            ec = stripepy_main(["--license"])

        assert f.getvalue().startswith("MIT License")
        assert ec == 0

    @staticmethod
    def test_stripepy_cite():
        f = io.StringIO()
        with redirect_stdout(f):
            ec = stripepy_main(["--cite"])

        assert f.getvalue().startswith("@article{stripepy,")
        assert ec == 0

    @staticmethod
    def test_stripepy_help():
        with pytest.raises(SystemExit) as cmd:
            f = io.StringIO()
            with redirect_stdout(f):
                stripepy_main(["--help"])

            assert f.getvalue().startswith("usage: stripepy")
            assert cmd.value.code == 0

        with pytest.raises(SystemExit) as cmd:
            f = io.StringIO()
            with redirect_stdout(f):
                stripepy_main(["--help", "--foobar"])

            assert f.getvalue().startswith("usage: stripepy")
            assert cmd.value.code == 0

        with pytest.raises(SystemExit) as cmd:
            f = io.StringIO()
            with redirect_stdout(f):
                stripepy_main(["--help", "--cite"])

            assert f.getvalue().startswith("usage: stripepy")
            assert cmd.value.code == 0

    @staticmethod
    def test_stripepy_invalid():
        f = io.StringIO()
        with redirect_stderr(f):
            ec = stripepy_main(["--cite", "--license"])

        assert "mutually exclusive" in f.getvalue()
        assert ec != 0

        with pytest.raises(SystemExit) as cmd:
            stripepy_main(["--foobar"])
            assert cmd.value.code != 0
