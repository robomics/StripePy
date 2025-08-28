# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from stripepy.data_structures import Result, Stripe


@pytest.mark.unit
class TestResult:
    def test_ctor(self):
        res = Result("chr1", 123)

        assert res.empty
        assert res.chrom[0] == "chr1"
        assert res.chrom[1] == 123
        assert res.roi is None

    def test_setters(self):
        res = Result("chr1", 123)

        res.set_min_persistence(1.0)
        with pytest.raises(RuntimeError, match="has already been set"):
            res.set_min_persistence(1.0)

        res.set_roi({"foobar": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="has already been set"):
            res.set_roi({})

        assert res.roi is not None
        assert "foobar" in res.roi
        assert res.roi["foobar"] == [1, 2, 3]

        with pytest.raises(AttributeError, match='No attribute named "invalid name"'):
            res.set("invalid name", [], "LT")

        with pytest.raises(ValueError, match="Location should be UT or LT"):
            res.set("all_minimum_points", [], "invalid location")

        res.set("all_minimum_points", [1, 2, 3], "LT")
        res.set("all_minimum_points", [4, 5, 6], "UT")

        with pytest.raises(RuntimeError, match="has already been set"):
            res.set("all_minimum_points", [], "LT")

        assert (res.get("all_minimum_points", "LT") == [1, 2, 3]).all()
        assert (res.get("all_minimum_points", "UT") == [4, 5, 6]).all()

    def test_getters(self):
        res = Result("chr1", 123)

        res.set("all_minimum_points", [1, 2, 3], "LT")

        with pytest.raises(RuntimeError, match="Attribute .* is not set"):
            res.min_persistence  # noqa

        res.set_min_persistence(1.23)
        assert res.min_persistence == 1.23

        with pytest.raises(AttributeError, match='No attribute named "invalid name"'):
            res.get("invalid name", "LT")

        with pytest.raises(ValueError, match="Location should be UT or LT"):
            res.get("all_minimum_points", "invalid location")

        with pytest.raises(RuntimeError, match='Attribute "all_minimum_points" for "UT" is not set'):
            res.get("all_minimum_points", "UT")

        with pytest.raises(RuntimeError, match='Attribute "all_maximum_points" for "LT" is not set'):
            res.get("all_maximum_points", "LT")

        assert len(res.get("stripes", "LT")) == 0

        assert (res.get("all_minimum_points", "LT") == [1, 2, 3]).all()

    def test_stripe_getters(self):
        res = Result("chr1", 123)

        stripes = [Stripe(10, 1.23, where="upper_triangular")]
        stripes[0].set_vertical_bounds(5, 10)
        stripes[0].set_horizontal_bounds(8, 12)
        stripes[0].compute_biodescriptors(csr_matrix((15, 15), dtype=float))

        with pytest.raises(AttributeError, match='does not have an attribute named "foobar"'):
            res.get_stripes_descriptor("foobar", "UT")

        with pytest.raises(ValueError, match="Location should be UT or LT"):
            res.get_stripes_descriptor("seed", "foobar")

        res.set("stripes", stripes, "UT")

        assert len(res.get_stripe_geo_descriptors("LT")) == 0
        assert len(res.get_stripe_bio_descriptors("LT")) == 0

        df = res.get_stripe_geo_descriptors("UT")
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
        assert np.isclose(df["top_persistence"].iloc[0], 1.23)
        assert df["left_bound"].iloc[0] == 8
        assert df["right_bound"].iloc[0] == 12
        assert df["top_bound"].iloc[0] == 5
        assert df["bottom_bound"].iloc[0] == 10

        expected_columns = [
            "inner_mean",
            "outer_mean",
            "rel_change",
            "cfx_of_variation",
            "inner_std",
            "outer_lsum",
            "outer_rsum",
            "outer_lsize",
            "outer_rsize",
            "outer_lmean",
            "outer_rmean",
            "min",
            "q1",
            "q2",
            "q3",
            "max",
        ]

        df = res.get_stripe_bio_descriptors("UT")
        assert df.columns.tolist() == expected_columns
        assert len(df) == 1

        assert df["inner_mean"].iloc[0] == 0
        assert df["outer_mean"].iloc[0] == 0
        assert np.isnan(df["rel_change"].iloc[0])
        assert df["inner_std"].iloc[0] == 0
