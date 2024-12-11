# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib

import h5py
import pytest

from stripepy import main

testdir = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.end2end
class TestStripePyCall:
    @staticmethod
    def test_stripepy_call(tmpdir):
        testfile = testdir / "data" / "4DNFI9GMP2J8.mcool"
        resolution = 10_000

        if not testfile.exists():
            raise RuntimeError(
                f'unable to find file "{testfile}". Did you download the test files prior to running pytest?'
            )

        args = [
            "call",
            str(testfile),
            str(resolution),
            "--glob-pers-min",
            "0.10",
            "--loc-pers-min",
            "0.33",
            "--loc-trend-min",
            "0.25",
            "--output-folder",
            str(tmpdir),
        ]
        main(args)

        outfile = pathlib.Path(tmpdir) / testfile.stem / str(resolution) / "results.hdf5"

        assert outfile.is_file()
        assert h5py.File(outfile).attrs.get("format", "unknown") == "HDF5::StripePy"
