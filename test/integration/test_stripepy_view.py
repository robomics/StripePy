# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import io
import pathlib

import pandas as pd
import pytest

from stripepy import main

testdir = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.end2end
class TestStripePyView:
    @staticmethod
    def test_view():
        testfile = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        if not testfile.exists():
            raise RuntimeError(
                f'unable to find file "{testfile}". Did you download the test files prior to running pytest?'
            )

        args = ["view", str(testfile)]
        buff = io.StringIO()
        with contextlib.redirect_stdout(buff):
            main(args)

        buff.seek(0)
        cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
        df = pd.read_table(buff, names=cols)

        assert len(df.columns) == len(cols)
        assert len(df) == len(df[~df.isnull().values])
        assert len(df) == 19750
