# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import datetime
import itertools
import json
import math
import pathlib
import platform
import tarfile
from typing import Dict, Sequence

import hictkpy as htk
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_helpers_cool import generate_singleres_test_file

from stripepy.data_structures import Result, ResultFile

testdir = pathlib.Path(__file__).resolve().parent.parent.parent


def _pyarrow_avail() -> bool:
    try:
        import pyarrow

        return True
    except ImportError:
        return False


def _directory_is_empty(path) -> bool:
    path = pathlib.Path(path)
    assert path.is_dir()
    return next(path.iterdir(), None) is None  # noqa


@pytest.mark.unit
class TestResultFile:
    def test_ctor_v1(self, tmpdir):
        path = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        with ResultFile(path) as f:
            assert f.path == path

    def test_ctor_v2(self, tmpdir):
        path = testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5"
        with ResultFile(path) as f:
            assert f.path == path

    def test_ctor_v3(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        path = testdir / "data" / "results_4DNFI9GMP2J8_v3.hdf5"
        with ResultFile(path) as f:
            assert f.path == path

        with pytest.raises(OSError):
            ResultFile(tmpdir / "asdf123.hdf5")

        path = pathlib.Path(tmpdir) / "test.hdf5"
        chroms = {"chr1": 10}
        with ResultFile.create(path, "w", chroms=chroms, resolution=1) as f:
            assert f.path == path

        with pytest.raises(OSError):
            ResultFile.create(path, "w", chroms=chroms, resolution=1)

        with pytest.raises(RuntimeError, match="cannot append to a file that has already been finalized!"):
            ResultFile.create(path, "a", chroms=chroms, resolution=1)

        with pytest.raises(RuntimeError, match="Please use .* to open a file in write mode"):
            ResultFile(path, "w")

    @staticmethod
    def _test_properties(path: pathlib.Path, version: int):
        with ResultFile(path) as f:
            assert f.path == path
            assert f.assembly == "unknown"
            assert f.resolution == 10_000

            try:
                f.creation_date  # noqa
            except ValueError:
                pytest.fail("creation-date attribute is not a valid date")

            assert f.format == "HDF5::StripePy"
            assert f.format_url == "https://github.com/paulsengroup/StripePy"
            assert f.format_version == version
            assert f.generated_by.startswith("StripePy")

            try:
                f.metadata  # noqa
            except json.decoder.JSONDecodeError:
                pytest.fail("metadata attribute is not a valid JSON")

            assert f.normalization is None

            assert len(f.chromosomes) == 24
            assert "chr1" in f.chromosomes

    def test_properties_v1(self):
        TestResultFile._test_properties(
            testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5",
            1,
        )

    def test_properties_v2(self):
        TestResultFile._test_properties(
            testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5",
            2,
        )

    def test_properties_v3(self):
        TestResultFile._test_properties(
            testdir / "data" / "results_4DNFI9GMP2J8_v3.hdf5",
            3,
        )

    def test_getters_short_v1(self):
        path = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        with ResultFile(path) as f:
            assert math.isclose(f.get_min_persistence("chr1"), 0.05)
            with pytest.raises(KeyError):
                f.get_min_persistence("foobar")

            df = f.get(None, "pseudodistribution", "LT")

            assert "pseudodistribution" in df
            assert len(df) == 308_837

            df = f.get(None, "stripes", "LT")
            columns = [
                "chrom",
                "seed",
                "top_persistence",
                "left_bound",
                "right_bound",
                "top_bound",
                "bottom_bound",
                "inner_mean",
                "outer_mean",
                "rel_change",
                "inner_std",
                "cfx_of_variation",
            ]

            assert df.columns.tolist() == columns
            assert len(df) == 17_639

            with pytest.raises(KeyError):
                f.get("foobar", "pseudodistribution", "LT")

            with pytest.raises(KeyError):
                f.get("chr1", "foobar", "LT")

            with pytest.raises(ValueError):
                f.get("chr1", "pseudodistribution", "foobar")

    def test_getters_short_v2(self):
        path = testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5"
        with ResultFile(path) as f:
            assert math.isclose(f.get_min_persistence("chr1"), 0.05)
            with pytest.raises(KeyError):
                f.get_min_persistence("foobar")

            df = f.get(None, "pseudodistribution", "LT")

            assert "pseudodistribution" in df
            assert len(df) == 308_837

            df = f.get(None, "stripes", "LT")
            columns = [
                "chrom",
                "seed",
                "top_persistence",
                "left_bound",
                "right_bound",
                "top_bound",
                "bottom_bound",
                "inner_mean",
                "inner_std",
                "outer_lmean",
                "outer_rmean",
                "outer_mean",
                "min",
                "q1",
                "q2",
                "q3",
                "max",
                "rel_change",
                "cfx_of_variation",
            ]

            assert len(df.columns) == len(columns)
            assert (df.columns == columns).all()
            assert len(df) == 17_639

            with pytest.raises(KeyError):
                f.get("foobar", "pseudodistribution", "LT")

            with pytest.raises(KeyError):
                f.get("chr1", "foobar", "LT")

            with pytest.raises(ValueError):
                f.get("chr1", "pseudodistribution", "foobar")

    def test_getters_short_v3(self):
        path = testdir / "data" / "results_4DNFI9GMP2J8_v3.hdf5"
        with ResultFile(path) as f:
            assert math.isclose(f.get_min_persistence("chr1"), 0.04)
            with pytest.raises(KeyError):
                f.get_min_persistence("foobar")

            df = f.get(None, "pseudodistribution", "LT")

            assert "pseudodistribution" in df
            assert len(df) == 308_837

            df = f.get(None, "stripes", "LT")
            columns = [
                "chrom",
                "seed",
                "top_persistence",
                "left_bound",
                "right_bound",
                "top_bound",
                "bottom_bound",
                "inner_mean",
                "inner_std",
                "outer_lsum",
                "outer_lsize",
                "outer_rsum",
                "outer_rsize",
                "min",
                "q1",
                "q2",
                "q3",
                "max",
                "outer_lmean",
                "outer_rmean",
                "outer_mean",
                "rel_change",
                "cfx_of_variation",
            ]

            assert len(df.columns) == len(columns)
            assert (df.columns == columns).all()
            assert len(df) == 24_009

            with pytest.raises(KeyError):
                f.get("foobar", "pseudodistribution", "LT")

            with pytest.raises(KeyError):
                f.get("chr1", "foobar", "LT")

            with pytest.raises(ValueError):
                f.get("chr1", "pseudodistribution", "foobar")

    @staticmethod
    def _compare_file_attributes(h5: ResultFile, path_to_attributes: pathlib.Path):
        with path_to_attributes.open() as f:
            attributes = json.load(f)

        try:
            datetime.datetime.fromisoformat(attributes.pop("creation-date"))
        except ValueError:
            pytest.fail("creation-date attribute is not a valid date")

        assert attributes.pop("generated-by", "missing").startswith("StripePy")
        attributes["resolution"] = attributes.pop("bin-size")

        TestResultFile._compare_attributes(h5, attributes)

    @staticmethod
    def _compare_result_attributes(h5: Result, path_to_attributes: pathlib.Path):
        with path_to_attributes.open() as f:
            attributes = json.load(f)

        TestResultFile._compare_attributes(h5, attributes)

    @staticmethod
    def _compare_attributes(obj, attributes: Dict):
        for attribute, value in attributes.items():
            normalized_attribute = attribute.replace("-", "_")
            if isinstance(value, str) and value.lower() == "none":
                value = None
            try:
                if value is None:
                    assert getattr(obj, normalized_attribute) is None
                else:
                    assert getattr(obj, normalized_attribute) == value
            except AttributeError:
                pytest.fail(f'{type(obj)} object does not have an attribute named "{normalized_attribute}"')

    @staticmethod
    def _compare_table(result: Result, path: pathlib.Path):
        name = str(path.stem)
        if name.endswith("_ut"):
            location = "UT"
        elif name.endswith("_lt"):
            location = "LT"
            assert name.endswith("_lt")
        else:
            pytest.fail(f'Unable to infer location from path "{path}"')

        name = name[:-3]

        if name == "bio_descriptors":
            cols = ["inner_mean", "outer_mean", "rel_change", "inner_std"]
            found = result.get_stripe_bio_descriptors(location)[cols]  # noqa
        elif name == "geo_descriptors":
            found = result.get_stripe_geo_descriptors(location)  # noqa
        else:
            found = result.get(name, location)  # noqa

        if not isinstance(found, pd.DataFrame):
            found = pd.DataFrame({name: found})

        expected = pd.read_parquet(path)
        assert_frame_equal(expected, found, check_exact=False)

    @staticmethod
    def _compare_chromosome(result: Result, path_to_tables: pathlib.Path):
        assert result.chrom[0] == str(path_to_tables.name)

        TestResultFile._compare_result_attributes(result, path_to_tables / "attributes.json")

        for path in path_to_tables.iterdir():
            if path.suffix == ".parquet":
                TestResultFile._compare_table(result, path)
            elif path.suffix == ".missing":
                with pytest.raises(RuntimeError, match=r"Attribute .* is not set"):
                    TestResultFile._compare_table(result, path)

    @staticmethod
    def _test_getters_long(
        name: str,
        result_file: pathlib.Path,
        reference_tar: pathlib.Path,
        version: int,
        tmpdir,
    ):
        tmpdir = pathlib.Path(tmpdir)
        prefix = pathlib.Path("stripepy-call-result-tables") / name
        with tarfile.open(reference_tar) as tar:
            members = (tarinfo for tarinfo in tar.getmembers() if tarinfo.name.startswith(prefix.as_posix()))
            py_major, py_minor, _ = platform.python_version_tuple()
            if py_major == "3" and int(py_minor) < 12:
                tar.extractall(path=tmpdir, members=members)
            else:
                tar.extractall(path=tmpdir, members=members, filter="data")

        reference_dir = tmpdir / prefix

        with ResultFile(result_file) as h5:
            assert h5.format_version == version
            TestResultFile._compare_file_attributes(h5, reference_dir / "attributes.json")
            for path in reference_dir.iterdir():
                if path.is_dir():
                    chrom = path.name
                    TestResultFile._compare_chromosome(h5[chrom], path)

    @pytest.mark.skipif(not _pyarrow_avail(), reason="pyarrow is not available")
    def test_getters_long_v1(self, tmpdir):
        TestResultFile._test_getters_long(
            name="results_4DNFI9GMP2J8_v1",
            result_file=testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5",
            reference_tar=testdir / "data" / "stripepy-call-result-tables.tar.xz",
            version=1,
            tmpdir=tmpdir,
        )

    @pytest.mark.skipif(not _pyarrow_avail(), reason="pyarrow is not available")
    def test_getters_long_v2(self, tmpdir):
        TestResultFile._test_getters_long(
            name="results_4DNFI9GMP2J8_v2",
            result_file=testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5",
            reference_tar=testdir / "data" / "stripepy-call-result-tables.tar.xz",
            version=2,
            tmpdir=tmpdir,
        )

    @pytest.mark.skipif(not _pyarrow_avail(), reason="pyarrow is not available")
    def test_getters_long_v3(self, tmpdir):
        TestResultFile._test_getters_long(
            name="results_4DNFI9GMP2J8_v3",
            result_file=testdir / "data" / "results_4DNFI9GMP2J8_v3.hdf5",
            reference_tar=testdir / "data" / "stripepy-call-result-tables.tar.xz",
            version=3,
            tmpdir=tmpdir,
        )

    @staticmethod
    def _write_mock_result_to_file(
        f: ResultFile,
        chrom: str,
        resolution: int,
        points: Sequence[int],
        pseudodistribution: Sequence[float],
        persistence: Sequence[float],
    ):
        res = Result(chrom, resolution * len(points))
        res.set_min_persistence(1.23)

        for location in ("UT", "LT"):
            res.set("pseudodistribution", pseudodistribution, location)

            for key in (
                "all_minimum_points",
                "all_maximum_points",
                "persistent_minimum_points",
                "persistent_maximum_points",
            ):
                res.set(key, points, location)

            for key in (
                "persistence_of_all_minimum_points",
                "persistence_of_all_maximum_points",
                "persistence_of_minimum_points",
                "persistence_of_maximum_points",
            ):
                res.set(key, persistence, location)

        f.write_descriptors(res)

    @staticmethod
    def _compare_results(
        path: pathlib.Path,
        chroms: Dict[str, int],
        points: Sequence[int],
        pseudodistribution: Sequence[float],
        persistence: Sequence[float],
    ):

        with ResultFile(path) as f:
            assert f.chromosomes == chroms
            assert f.normalization == "weight"
            assert f.metadata.get("key", "missing") == "value"

            for chrom, location in itertools.product(chroms, ("UT", "LT")):
                assert np.isclose(f.get_min_persistence(chrom), 1.23)
                assert np.allclose(
                    f.get(chrom, "pseudodistribution", location)["pseudodistribution"], pseudodistribution
                )

                for key in ("all_minimum_points", "all_maximum_points"):
                    assert (f.get(chrom, key, location)[key] == points).all()

                for key in ("persistence_of_all_minimum_points", "persistence_of_all_maximum_points"):
                    assert np.allclose(f.get(chrom, key, location)[key], persistence)

    def test_file_creation(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        resolution = 10_000
        clr_file = generate_singleres_test_file(tmpdir / "test.cool", resolution)
        chroms = htk.File(clr_file).chromosomes()

        points = [1, 2, 3]
        persistence = [4.0, 5.0, 6.0]
        pseudodistribution = [7.0, 8.0, 9.0]

        # Create a mock ResultFile
        path = tmpdir / "results.hdf5"
        with ResultFile.create_from_file(
            path,
            "w",
            htk.File(clr_file),
            normalization="weight",
            metadata={"key": "value"},
        ) as f:
            for chrom in chroms:
                TestResultFile._write_mock_result_to_file(f, chrom, resolution, points, pseudodistribution, persistence)

        TestResultFile._compare_results(path, chroms, points, pseudodistribution, persistence)

    def test_progressive_file_creation(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        points = [1, 2, 3]
        persistence = [4.0, 5.0, 6.0]
        pseudodistribution = [7.0, 8.0, 9.0]

        resolution = 10_000
        clr_file = generate_singleres_test_file(
            tmpdir / "test.cool", resolution, {"chr1": len(points) * resolution, "chr2": len(points) * resolution}
        )
        chroms = htk.File(clr_file).chromosomes()

        path = tmpdir / "results.hdf5"
        with ResultFile.create(path, "a", chroms, resolution, normalization="weight", metadata={"key": "value"}) as f:
            pass

        for chrom in chroms:
            with ResultFile.append(path) as f:
                TestResultFile._write_mock_result_to_file(f, chrom, resolution, points, pseudodistribution, persistence)

        with ResultFile.append(path) as f:
            f.finalize()

        TestResultFile._compare_results(path, chroms, points, pseudodistribution, persistence)
