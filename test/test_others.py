import functools
import gc
import pathlib
import tempfile
from typing import Dict, List, Union

import hictkpy as htk
import pandas as pd
import pytest

from stripepy.others import cmap_loading


@functools.cache
def _generate_chromosomes() -> Dict[str, int]:
    return {
        "chr2L": 23513712,
        "chr2R": 25286936,
        "chr3L": 28110227,
        "chr3R": 32079331,
        "chr4": 1348131,
        "chrX": 23542271,
        "chrY": 3667352,
        "chrM": 19524,
    }


def _generate_singleres_test_file(
    path: pathlib.Path, resolution: int, chromosomes: Union[Dict[str, int], None] = None
) -> pathlib.Path:
    if chromosomes is None:
        chromosomes = _generate_chromosomes()

    # TODO remove str() after upgrading to hictkpy v1.0.0
    writer = htk.cooler.FileWriter(str(path), resolution=resolution, chromosomes=chromosomes)
    writer.add_pixels(pd.DataFrame({"bin1_id": [0], "bin2_id": [0], "count": 1}))
    writer.finalize()

    del writer
    gc.collect()

    return pathlib.Path(path)


def _discretize(v: List[int], factor: int) -> List[int]:
    assert factor > 0
    return [(x + factor - 1) // factor for x in v]


class TestCmapLoading:
    # TODO revise error messages
    def test_invalid_paths(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        with pytest.raises(Exception, match="Unsupported file format:"):
            cmap_loading(tmpdir / "foo", 5000)

    def test_invalid_formats(self, tmpdir):
        invalid_file = pathlib.Path(__file__).resolve()
        folder = pathlib.Path(tmpdir)
        with pytest.raises(Exception, match="Unsupported file format:"):
            cmap_loading(invalid_file, 5000)

        with pytest.raises(Exception, match="Unsupported file format:"):
            cmap_loading(folder, 5000)

    def test_invalid_resolutions(self, tmpdir):
        invalid_resolutions = [-1, 0, 1000.0, 5000]
        with tempfile.NamedTemporaryFile(dir=tmpdir) as clr:
            clr.close()
            path_to_clr = _generate_singleres_test_file(pathlib.Path(clr.name), 1000)

            for res in invalid_resolutions:
                with pytest.raises(Exception):
                    cmap_loading(path_to_clr, res)

    def test_valid_files(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        chromosomes = {"A": 100_000, "B": 50_000, "C": 10_000}
        resolution = 1000
        path_to_clr = _generate_singleres_test_file(tmpdir / "test.cool", resolution, chromosomes)

        # TODO remove str() after upgrading to hictkpy v1.0.0
        f, starts, ends, sizes = cmap_loading(str(path_to_clr), resolution)
        assert f.resolution() == 1000
        assert len(starts) == len(chromosomes)
        assert len(starts) == len(ends)
        assert len(starts) == len(sizes)

        print(starts)
        print(ends)
        print(sizes)

        assert starts == _discretize([0, 100_000, 150_000], resolution)
        assert ends == _discretize([100_000, 150_000, 160_000], resolution)
        assert sizes == list(chromosomes.values())

        # TODO enable after merging https://github.com/paulsengroup/StripePy/pull/4
        # with pytest.raises(Exception):
        #     cmap_loading(path_to_clr, resolution + 1)
