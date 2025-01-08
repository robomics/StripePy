# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib

import h5py
import hictkpy
import pytest

from stripepy import main

from .common import matplotlib_avail

testdir = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.end2end
class TestStripePyCall:
    @staticmethod
    def setup_class():
        test_files = [
            testdir / "data" / "4DNFI9GMP2J8.mcool",
        ]

        for f in test_files:
            if not f.exists():
                raise RuntimeError(
                    f'unable to find file "{f}". Did you download the test files prior to running pytest?'
                )

    @staticmethod
    def test_stripepy_call(tmpdir):
        testfile = testdir / "data" / "4DNFI9GMP2J8.mcool"
        resolution = 10_000

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

    @staticmethod
    def test_stripepy_call_with_roi(tmpdir):
        testfile = testdir / "data" / "4DNFI9GMP2J8.mcool"
        resolution = 10_000

        chrom_size_cutoff = max(hictkpy.MultiResFile(testfile).chromosomes().values()) - 1

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
            "--min-chrom-size",
            str(chrom_size_cutoff),
            "--roi",
            "middle",
        ]
        if not matplotlib_avail():
            with pytest.raises(ImportError):
                main(args)
            pytest.skip("matplotlib not available")

        main(args)

        outfile = pathlib.Path(tmpdir) / testfile.stem / str(resolution) / "results.hdf5"

        assert outfile.is_file()
        assert h5py.File(outfile).attrs.get("format", "unknown") == "HDF5::StripePy"
