# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import gc
import multiprocessing as mp
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as ss


def _ctor(m: Union[ss.csr_matrix, ss.csc_matrix]):
    return (
        (mp.RawArray(np.ctypeslib.as_ctypes_type(m.data.dtype), len(m.data)), m.data.dtype),
        (mp.RawArray(np.ctypeslib.as_ctypes_type(m.indices.dtype), len(m.indices)), m.indices.dtype),
        (mp.RawArray(np.ctypeslib.as_ctypes_type(m.indptr.dtype), len(m.indptr)), m.indptr.dtype),
        m.shape,
    )


def _copy_data(src: npt.NDArray, dest: mp.RawArray, dtype: npt.DTypeLike):
    np.copyto(np.frombuffer(dest, dtype=dtype), src, casting="safe")


class SharedCSRMatrix(object):
    def __init__(self, m: ss.csr_matrix):
        self._data, self._indices, self._indptr, self._shape = _ctor(m)

        _copy_data(m.data, *self._data)
        _copy_data(m.indices, *self._indices)
        _copy_data(m.indptr, *self._indptr)

    def get(self) -> ss.csr_matrix:
        return ss.csr_matrix(
            (
                np.frombuffer(self._data[0], dtype=self._data[1]),
                np.frombuffer(self._indices[0], dtype=self._indices[1]),
                np.frombuffer(self._indptr[0], dtype=self._indptr[1]),
            ),
            shape=self._shape,
            copy=False,
        )


class SharedCSCMatrix(object):
    def __init__(self, m: ss.csc_matrix):
        self._data, self._indices, self._indptr, self._shape = _ctor(m)

        _copy_data(m.data, *self._data)
        _copy_data(m.indices, *self._indices)
        _copy_data(m.indptr, *self._indptr)

    def get(self) -> ss.csc_matrix:
        return ss.csc_matrix(
            (
                np.frombuffer(self._data[0], dtype=self._data[1]),
                np.frombuffer(self._indices[0], dtype=self._indices[1]),
                np.frombuffer(self._indptr[0], dtype=self._indptr[1]),
            ),
            shape=self._shape,
            copy=False,
        )


class SharedSparseMatrix(object):
    def __init__(self, m):
        if isinstance(m, ss.csr_matrix):
            self._m = SharedCSRMatrix(m)
        elif isinstance(m, ss.csc_matrix):
            self._m = SharedCSCMatrix(m)
        else:
            self._m = SharedSparseMatrix(ss.csr_matrix(m))

    def get(self) -> Union[ss.csr_matrix, ss.csc_matrix]:
        return self._m.get()


def set_shared_state(lt_matrix: SharedSparseMatrix, ut_matrix: SharedSparseMatrix):
    global _lower_triangular_matrix
    global _upper_triangular_matrix

    _lower_triangular_matrix = lt_matrix
    _upper_triangular_matrix = ut_matrix


def unset_shared_state():
    global _lower_triangular_matrix
    global _upper_triangular_matrix

    del _lower_triangular_matrix
    del _upper_triangular_matrix
    gc.collect()


def get_shared_state(key: str):
    if key in {"lower", "LT"}:
        global _lower_triangular_matrix
        return _lower_triangular_matrix

    if key in {"upper", "UT"}:
        global _upper_triangular_matrix
        return _upper_triangular_matrix

    raise ValueError("Invalid key")
