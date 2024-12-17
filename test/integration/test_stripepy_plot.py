# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib

import pytest

from stripepy import main

testdir = pathlib.Path(__file__).resolve().parent.parent


def compare_images(expected: pathlib.Path, actual: pathlib.Path, tol: float = 0.1):
    import matplotlib.testing.compare

    res = matplotlib.testing.compare.compare_images(str(expected), str(actual), tol)
    if res is not None:
        pytest.fail(res)


@pytest.mark.end2end
class TestStripePyPlot:
    @staticmethod
    def setup_class():
        test_files = [
            testdir / "data" / "4DNFI9GMP2J8.mcool",
            testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5",
            testdir / "data" / "contact_map.png",
            testdir / "data" / "contact_map_with_seeds.png",
            testdir / "data" / "contact_map_with_stripes.png",
            testdir / "data" / "contact_map_with_stripes_no_heights.png",
            testdir / "data" / "pseudodistribution.png",
            testdir / "data" / "stripe_hist.png",
            testdir / "data" / "stripe_hist_gw.png",
        ]

        for f in test_files:
            if not f.exists():
                raise RuntimeError(
                    f'unable to find file "{f}". Did you download the test files prior to running pytest?'
                )

    @staticmethod
    def test_contact_map(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        matrix_file = testdir / "data" / "4DNFI9GMP2J8.mcool"
        expected = testdir / "data" / "contact_map.png"

        resolution = 10_000
        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = ["plot", "contact-map", str(matrix_file), str(resolution), str(outfile), "--region", region]
        main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_contact_map_with_seeds(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        matrix_file = testdir / "data" / "4DNFI9GMP2J8.mcool"
        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = testdir / "data" / "contact_map_with_seeds.png"

        resolution = 10_000
        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = [
            "plot",
            "contact-map",
            str(matrix_file),
            str(resolution),
            str(outfile),
            "--stripepy-hdf5",
            str(stripe_file),
            "--highlight-seeds",
            "--region",
            region,
        ]
        main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_contact_map_with_stripes(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        matrix_file = testdir / "data" / "4DNFI9GMP2J8.mcool"
        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = testdir / "data" / "contact_map_with_stripes.png"

        resolution = 10_000
        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = [
            "plot",
            "contact-map",
            str(matrix_file),
            str(resolution),
            str(outfile),
            "--stripepy-hdf5",
            str(stripe_file),
            "--highlight-stripes",
            "--region",
            region,
        ]
        main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_contact_map_with_stripes_no_heights(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        matrix_file = testdir / "data" / "4DNFI9GMP2J8.mcool"
        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = testdir / "data" / "contact_map_with_stripes_no_heights.png"

        resolution = 10_000
        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = [
            "plot",
            "contact-map",
            str(matrix_file),
            str(resolution),
            str(outfile),
            "--stripepy-hdf5",
            str(stripe_file),
            "--highlight-stripes",
            "--ignore-stripe-heights",
            "--region",
            region,
        ]
        main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_pseudodistribution(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = testdir / "data" / "pseudodistribution.png"

        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = ["plot", "pseudodistribution", str(stripe_file), str(outfile), "--region", region]
        main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_stripe_hist(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = testdir / "data" / "stripe_hist.png"

        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = ["plot", "stripe-hist", str(stripe_file), str(outfile), "--region", region]
        main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_stripe_hist_gw(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = testdir / "data" / "stripe_hist_gw.png"

        outfile = tmpdir / "img.png"
        args = ["plot", "stripe-hist", str(stripe_file), str(outfile)]
        main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)
