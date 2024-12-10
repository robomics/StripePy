# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import shutil
import tempfile

import pytest
from scipy.sparse import csr_matrix

from stripepy.IO import Result, create_folders_for_plots, remove_and_create_folder
from stripepy.utils.stripe import Stripe


def _directory_is_empty(path) -> bool:
    path = pathlib.Path(path)
    assert path.is_dir()
    return next(path.iterdir(), None) is None  # noqa


@pytest.mark.unit
def test_folders_for_plots(tmpdir):
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
        test_paths = [
            pathlib.Path(tmpdir),
            pathlib.Path(tmpdir) / "dir2",
        ]

        for test_dir in test_paths:
            if test_dir.exists():
                shutil.rmtree(test_dir)

            assert create_folders_for_plots(test_dir) == [
                test_dir,
                test_dir / "1_preprocessing",
                test_dir / "2_TDA",
                test_dir / "3_shape_analysis",
                test_dir / "4_biological_analysis",
                test_dir / "3_shape_analysis" / "local_pseudodistributions",
            ]


@pytest.mark.unit
class TestRemoveAndCreateFolder:
    # RuntimeError(f"output folder {path} already exists. Pass --force to overwrite it.")
    @staticmethod
    def test_create_new_folder(tmpdir):
        with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            test_dir = tmpdir / "out"
            remove_and_create_folder(test_dir, force=True)
            assert test_dir.is_dir()
            assert _directory_is_empty(test_dir)

            with pytest.raises(RuntimeError, match="already exists. Pass --force to overwrite it"):
                remove_and_create_folder(test_dir, force=False)

    @staticmethod
    def test_overwrite_existing_folder(tmpdir):
        with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
            tmpdir = pathlib.Path(tmpdir)

            test_dir = tmpdir / "out"

            (test_dir / "dir").mkdir(parents=True)
            (test_dir / "file.txt").touch()

            assert not _directory_is_empty(test_dir)
            remove_and_create_folder(test_dir, force=True)
            assert _directory_is_empty(test_dir)


@pytest.mark.unit
def test_create_folders_for_plots(tmpdir):
    with tempfile.TemporaryDirectory(dir=tmpdir) as tmpdir:
        tmpdir = pathlib.Path(tmpdir)

        test_dir = tmpdir / "out"
        result = create_folders_for_plots(test_dir)
        assert isinstance(result, list)
        assert test_dir.is_dir()


@pytest.mark.unit
class TestResult:
    def test_ctor(self):
        res = Result("chr1")

        assert res.empty
        assert res.chrom == "chr1"
        assert res.roi is None

    def test_setters(self):
        res = Result("chr1")

        res.set_min_persistence(1.0)
        with pytest.raises(RuntimeError, match="has already been set"):
            res.set_min_persistence(1.0)

        res.set_roi({"foobar": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="has already been set"):
            res.set_roi({})

        assert res.roi is not None
        assert "foobar" in res.roi
        assert res.roi["foobar"] == [1, 2, 3]

        with pytest.raises(RuntimeError, match='No attribute named "invalid name"'):
            res.set("invalid name", [], "LT")

        with pytest.raises(RuntimeError, match="Location should be UT or LT"):
            res.set("all_minimum_points", [], "invalid location")

        res.set("all_minimum_points", [1, 2, 3], "LT")
        res.set("all_minimum_points", [4, 5, 6], "UT")

        with pytest.raises(RuntimeError, match="has already been set"):
            res.set("all_minimum_points", [], "LT")

        assert (res.get("all_minimum_points", "LT") == [1, 2, 3]).all()
        assert (res.get("all_minimum_points", "UT") == [4, 5, 6]).all()

    def test_getters(self):
        res = Result("chr1")

        res.set("all_minimum_points", [1, 2, 3], "LT")

        with pytest.raises(RuntimeError, match="Attribute .* is not set"):
            res.min_persistence  # noqa

        res.set_min_persistence(1.23)
        assert res.min_persistence == 1.23

        with pytest.raises(RuntimeError, match='No attribute named "invalid name"'):
            res.get("invalid name", "LT")

        with pytest.raises(RuntimeError, match="Location should be UT or LT"):
            res.get("all_minimum_points", "invalid location")

        with pytest.raises(RuntimeError, match='Attribute "all_minimum_points" for "UT" is not set'):
            res.get("all_minimum_points", "UT")

        with pytest.raises(RuntimeError, match='Attribute "all_maximum_points" for "LT" is not set'):
            res.get("all_maximum_points", "LT")

        assert len(res.get("stripes", "LT")) == 0

        assert (res.get("all_minimum_points", "LT") == [1, 2, 3]).all()

    def test_stripe_getters(self):
        res = Result("chr1")

        stripes = [Stripe(10, 1.23)]
        stripes[0].set_vertical_bounds(5, 10)
        stripes[0].set_horizontal_bounds(8, 12)
        stripes[0].compute_biodescriptors(csr_matrix((15, 15), dtype=float))

        with pytest.raises(RuntimeError, match='does not have an attribute named "foobar"'):
            res.get_stripes_descriptor("foobar", "UT")

        with pytest.raises(RuntimeError, match="Location should be UT or LT"):
            res.get_stripes_descriptor("seed", "foobar")

        res.set("stripes", stripes, "LT")

        assert len(res.get_stripe_geo_descriptors("UT")) == 0
        assert len(res.get_stripe_bio_descriptors("UT")) == 0

        df = res.get_stripe_geo_descriptors("LT")
        assert df.columns.tolist() == [
            "seed",
            "top_persistence",
            "left_bound",
            "right_bound",
            "top_bound",
            "bottom_bound",
        ]
        assert len(df) == 1

        assert df["seed"].iloc[0] == 10
        assert df["top_persistence"].iloc[0] == 1.23
        assert df["left_bound"].iloc[0] == 8
        assert df["right_bound"].iloc[0] == 12
        assert df["top_bound"].iloc[0] == 5
        assert df["bottom_bound"].iloc[0] == 10

        df = res.get_stripe_bio_descriptors("LT")
        assert df.columns.tolist() == ["inner_mean", "outer_mean", "rel_change", "inner_std"]
        assert len(df) == 1

        assert df["inner_mean"].iloc[0] == 0
        assert df["outer_mean"].iloc[0] == 0
        assert df["rel_change"].iloc[0] == -1
        assert df["inner_std"].iloc[0] == 0
