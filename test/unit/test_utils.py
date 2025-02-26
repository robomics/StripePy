# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import pytest

from stripepy.utils import define_region_of_interest


@pytest.mark.unit
class TestDefineRegionOfInterest:
    def test_middle_large_chromosome(self):
        roi = define_region_of_interest(location="middle", chrom_size=1000, resolution=10, window_size=500)
        assert roi == {"genomic": (250, 750), "matrix": (25, 75)}

    def test_middle_small_chromosome(self):
        roi = define_region_of_interest(location="middle", chrom_size=123, resolution=10, window_size=500)
        assert roi == {"genomic": (0, 123), "matrix": (0, 12)}

    def test_start_large_chromosome(self):
        roi = define_region_of_interest(location="start", chrom_size=1000, resolution=10, window_size=500)
        assert roi == {"genomic": (0, 500), "matrix": (0, 50)}

    def test_start_small_chromosome(self):
        roi = define_region_of_interest(location="start", chrom_size=123, resolution=10, window_size=500)
        assert roi == {"genomic": (0, 123), "matrix": (0, 12)}

    def test_noop(self):
        assert define_region_of_interest(location=None, chrom_size=0, resolution=0) is None
        assert define_region_of_interest(location="middle", chrom_size=1000, resolution=10, window_size=0) is None

    def test_invalid_params(self):
        with pytest.raises(Exception):
            define_region_of_interest(location="middle", chrom_size=0, resolution=10, window_size=500)
            define_region_of_interest(location="middle", chrom_size=1000, resolution=0, window_size=500)

            define_region_of_interest(location="foo", chrom_size=1000, resolution=10, window_size=500)
