import numpy as np
import scipy.sparse as ss

from stripepy.stripepy import log_transform


class TestLogTransform:

    def test_empty(self):
        I = ss.csr_matrix((0, 0), dtype=float)
        Iproc = log_transform(I)

        assert Iproc.shape == (0, 0)
        # assert isinstance(Iproc.dtype, np.floating) # TODO robomics fix me!

    def test_all_finite(self):
        I = ss.rand(100, 100, density=0.5, format="csr")
        Iproc = log_transform(I)

        assert I.size == Iproc.size
        assert I.shape == Iproc.shape
        assert np.isfinite(Iproc.data).all()

    def test_with_nans(self):
        I = ss.rand(100, 100, density=0.5, format="csr")
        size_I = I.size
        mean_I = I.mean()
        num_nan_values = (I.data >= mean_I).sum()
        I.data[I.data >= mean_I] = np.nan

        Iproc = log_transform(I)

        assert np.isfinite(Iproc.data).all()
        assert Iproc.size == size_I - num_nan_values
