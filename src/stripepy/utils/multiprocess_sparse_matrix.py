# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import gc
import multiprocessing as mp
import platform
import time
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as ss

from .common import pretty_format_elapsed_time


class _SharedRawArrayWrapper(object):
    def __init__(self, data: npt.NDArray, capacity: Optional[int] = None):
        if capacity is None:
            capacity = len(data)

        self._dtype = data.dtype
        self._size = len(data)
        self._capacity = capacity
        self._data = mp.RawArray(np.ctypeslib.as_ctypes_type(self._dtype), self._capacity)

        self.assign(data)

    def __len__(self) -> int:
        return self._size

    def can_assign(self, data: npt.NDArray) -> bool:
        if len(data) > self._capacity:
            return False

        if data.dtype != self._dtype:
            return False

        return True

    def assign(self, data: npt.NDArray):
        assert self.can_assign(data)
        self._size = len(data)
        np.copyto(np.frombuffer(self._data, dtype=data.dtype, count=self._size), data, casting="safe")

    @property
    def dtype(self) -> npt.DTypeLike:
        return self._dtype

    @property
    def capacity(self) -> int:
        return self._size

    @property
    def data(self) -> npt.NDArray:
        return np.frombuffer(self._data, count=self._size, dtype=self._dtype)

    @property
    def raw_data(self) -> mp.RawArray:
        return self._data


class _SharedSparseMatrixBase(object):
    def __init__(self, m: Union[ss.csr_matrix, ss.csc_matrix], logger=None):
        assert isinstance(m, ss.csr_matrix) or isinstance(m, ss.csc_matrix)

        t0 = time.time()
        self._data = _SharedRawArrayWrapper(m.data)
        self._indices = _SharedRawArrayWrapper(m.indices)
        self._indptr = _SharedRawArrayWrapper(m.indptr)
        self._shape = m.shape
        self._matrix_type_str = "CSR" if isinstance(m, ss.csr_matrix) else "CSC"
        self._matrix_type = type(m)

        if logger:
            logger.debug(
                "allocation and initialization of a %s matrix (%d nnz) in shared memory took %s",
                self._matrix_type_str,
                m.nnz,
                pretty_format_elapsed_time(t0),
            )

    def get(self) -> Union[ss.csr_matrix, ss.csc_matrix]:

        return self._matrix_type(
            (self._data.data, self._indices.data, self._indptr.data),
            shape=self._shape,
            copy=False,
        )

    def can_assign(self, m: Union[ss.csr_matrix, ss.csc_matrix]) -> bool:
        assert isinstance(m, self._matrix_type)

        if platform.system() not in {"Darwin", "Windows"}:
            return False
        return all(
            (
                self._data.can_assign(m.data),
                self._indices.can_assign(m.indices),
                self._indptr.can_assign(m.indptr),
            )
        )

    def assign(self, m: Union[ss.csr_matrix, ss.csc_matrix], logger=None):
        t0 = time.time()
        self._data.assign(m.data)
        self._indices.assign(m.indices)
        self._indptr.assign(m.indptr)
        self._shape = m.shape
        if logger:
            logger.debug(
                "assigning to a %s matrix (%d nnz) in shared memory took %s",
                self._matrix_type_str,
                m.nnz,
                pretty_format_elapsed_time(t0),
            )


class SharedCSRMatrix(_SharedSparseMatrixBase):
    def __init__(self, m: ss.csr_matrix, logger=None):
        assert isinstance(m, ss.csr_matrix)
        super().__init__(m, logger)


class SharedCSCMatrix(_SharedSparseMatrixBase):
    def __init__(self, m: ss.csc_matrix, logger=None):
        assert isinstance(m, ss.csc_matrix)
        super().__init__(m, logger)


class SharedSparseMatrix(object):
    def __init__(self, m, logger=None):
        if isinstance(m, ss.csr_matrix):
            self._m = SharedCSRMatrix(m, logger)
        elif isinstance(m, ss.csc_matrix):
            self._m = SharedCSCMatrix(m, logger)
        else:
            self._m = SharedSparseMatrix(ss.csr_matrix(m), logger)

    def get(self) -> Union[ss.csr_matrix, ss.csc_matrix]:
        return self._m.get()

    def can_assign(self, m) -> bool:
        if isinstance(m, ss.csr_matrix):
            return isinstance(self._m, SharedCSRMatrix) and self._m.can_assign(m)
        if isinstance(m, ss.csc_matrix):
            return isinstance(self._m, SharedCSCMatrix) and self._m.can_assign(m)

        raise NotImplementedError

    def assign(self, m, logger=None):
        assert self.can_assign(m) and self._m.can_assign(m)
        self._m.assign(m, logger)


def set_shared_state(lt_matrix: SharedSparseMatrix, ut_matrix: SharedSparseMatrix):
    if lt_matrix is not None:
        global _lower_triangular_matrix
        _lower_triangular_matrix = lt_matrix
    if ut_matrix is not None:
        global _upper_triangular_matrix
        _upper_triangular_matrix = ut_matrix


def unset_shared_state():
    call_gc = False
    if "_lower_triangular_matrix" in globals():
        global _lower_triangular_matrix
        del _lower_triangular_matrix
        call_gc = True

    if "_upper_triangular_matrix" in globals():
        global _upper_triangular_matrix
        del _upper_triangular_matrix
        call_gc = True

    if call_gc:
        gc.collect()


def get_shared_state(key: str):
    if key in {"lower", "LT"}:
        global _lower_triangular_matrix
        return _lower_triangular_matrix

    if key in {"upper", "UT"}:
        global _upper_triangular_matrix
        return _upper_triangular_matrix

    raise ValueError("Invalid key")
