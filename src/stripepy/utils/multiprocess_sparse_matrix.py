# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import gc
import multiprocessing as mp
import time
from typing import Dict, Optional, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as ss

from .common import pretty_format_elapsed_time

SparseMatrix = Union[ss.csr_matrix, ss.csc_matrix]


class _SharedRawArrayWrapper(object):
    def __init__(self, data: npt.NDArray, capacity: Optional[int] = None):
        if capacity is None:
            capacity = len(data)

        assert capacity >= len(data)

        self._dtype = data.dtype
        self._capacity = capacity
        self._shared_buffer = mp.RawArray(np.ctypeslib.as_ctypes_type(self._dtype), self._capacity)
        self._numpy_array = None

        self.assign(data)

    def __len__(self) -> int:
        return len(self._numpy_array)

    def can_assign(self, data: npt.NDArray) -> bool:
        if len(data) > self._capacity:
            return False

        if data.dtype != self._dtype:
            return False

        return True

    def assign(self, data: npt.NDArray):
        assert self.can_assign(data)
        new_size = len(data)
        self._numpy_array = np.frombuffer(self._shared_buffer, dtype=self._dtype, count=new_size)
        np.copyto(self._numpy_array, data, casting="safe")

    def shrink(self, new_size: int):
        if new_size == len(self._numpy_array):
            return
        assert new_size < self._capacity
        self._numpy_array = np.frombuffer(self._shared_buffer, dtype=self._dtype, count=new_size)

    @property
    def dtype(self) -> npt.DTypeLike:
        return self._dtype

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def data(self) -> npt.NDArray:
        return self._numpy_array

    @property
    def raw_data(self) -> mp.RawArray:
        return self._shared_buffer


class _SharedTriangularSparseMatrixBase(object):
    def __init__(self, chrom: str, m: SparseMatrix, logger=None, max_nnz: Optional[int] = None):
        assert isinstance(m, ss.csr_matrix) or isinstance(m, ss.csc_matrix)

        if max_nnz is None:
            capacity_data = len(m.data)
            capacity_indices = len(m.indices)
            capacity_indptr = len(m.indptr)
        else:
            spare_capacity_pct = max(1.0, max_nnz / m.nnz)
            capacity_data = int(np.ceil(len(m.data) * spare_capacity_pct))
            capacity_indices = int(np.ceil(len(m.indices) * spare_capacity_pct))
            capacity_indptr = int(np.ceil(len(m.indptr) * spare_capacity_pct))

        t0 = time.time()
        self._chrom = chrom
        self._data = _SharedRawArrayWrapper(m.data, capacity_data)
        self._indices = _SharedRawArrayWrapper(m.indices, capacity_indices)
        self._indptr = _SharedRawArrayWrapper(m.indptr, capacity_indptr)
        self._shape = m.shape
        self._matrix_type_str = "CSR" if isinstance(m, ss.csr_matrix) else "CSC"
        self._matrix_type = type(m)
        self._sum = None  # self.get().sum()

        if logger:
            logger.debug(
                "allocation and initialization of a %s matrix (%s; %d nnz) in shared memory took %s",
                self._matrix_type_str,
                self._shape,
                m.nnz,
                pretty_format_elapsed_time(t0),
            )

    def get(self) -> SparseMatrix:
        return self._matrix_type(
            (self._data.data, self._indices.data, self._indptr.data),
            shape=self._shape,
            copy=False,
        )

    def can_assign(self, m: SparseMatrix) -> bool:
        assert isinstance(m, self._matrix_type)

        return all(
            (
                self._data.can_assign(m.data),
                self._indices.can_assign(m.indices),
                self._indptr.can_assign(m.indptr),
            )
        )

    def assign(self, chrom: str, m: SparseMatrix, logger=None):
        assert self.can_assign(m)

        t0 = time.time()
        self._chrom = chrom
        self._data.assign(m.data)
        self._indices.assign(m.indices)
        self._indptr.assign(m.indptr)
        self._shape = m.shape
        self._sum = None  # self.get().sum()

        if logger:
            logger.debug(
                "assigning to a %s matrix (%s; %d nnz) in shared memory took %s",
                self._matrix_type_str,
                self._chrom,
                m.nnz,
                pretty_format_elapsed_time(t0),
            )

    def update_metadata(self, chrom, shape, data_shape, indices_shape, indptr_shape, sum):
        if chrom == self._chrom:
            return
        old_data_len = len(self._data)
        old_indices_len = len(self._indices)
        old_indptr_len = len(self._indptr)
        try:
            self._data.shrink(data_shape)
            self._indices.shrink(indices_shape)
            self._indptr.shrink(indptr_shape)
        except Exception:  # noqa
            self._data.shrink(old_data_len)
            self._indices.shrink(old_indices_len)
            self._indptr.shrink(old_indptr_len)
            raise

        self._chrom = chrom
        self._shape = shape
        if sum is not None:
            assert np.isclose(self.get().sum(), sum)

    @property
    def chrom(self) -> str:
        return self._chrom

    @property
    def metadata(self) -> Dict:
        return {
            "chrom": self._chrom,
            "shape": self._shape,
            "data_shape": len(self._data),
            "indices_shape": len(self._indices),
            "indptr_shape": len(self._indptr),
            "sum": self._sum,
        }


class SharedTriangularCSRMatrix(_SharedTriangularSparseMatrixBase):
    def __init__(self, chrom: str, m: ss.csr_matrix, logger=None, max_nnz: Optional[int] = None):
        assert isinstance(m, ss.csr_matrix)
        super().__init__(chrom, m, logger, max_nnz)

    @staticmethod
    def _from_shared_buffers(
        chrom: str,
        data: _SharedRawArrayWrapper,
        indices: _SharedRawArrayWrapper,
        indptr: _SharedRawArrayWrapper,
        shape,
    ):
        assert shape[0] == shape[1]
        m = SharedTriangularCSRMatrix(chrom, ss.csr_matrix([], shape=(0, 0), dtype=data.dtype))
        m._data = data
        m._indices = indices
        m._indptr = indptr
        m._shape = shape
        m._matrix_type_str = "CSR"
        m._matrix_type = ss.csr_matrix

        # Assertion commented out for performance reasons
        # assert ss.triu(m.get(), k=1).sum() == 0 or ss.tril(m.get(), k=-1).sum() == 0

        return m

    @property
    def T(self):  # noqa
        return SharedTriangularCSCMatrix._from_shared_buffers(  # noqa
            self._chrom,
            self._data,
            self._indices,
            self._indptr,
            self._shape,
        )


class SharedTriangularCSCMatrix(_SharedTriangularSparseMatrixBase):
    def __init__(self, chrom: str, m: ss.csc_matrix, logger=None, max_nnz: Optional[int] = None):
        assert isinstance(m, ss.csc_matrix)
        super().__init__(chrom, m, logger, max_nnz)

    @staticmethod
    def _from_shared_buffers(
        chrom: str,
        data: _SharedRawArrayWrapper,
        indices: _SharedRawArrayWrapper,
        indptr: _SharedRawArrayWrapper,
        shape,
    ):
        assert shape[0] == shape[1]
        m = SharedTriangularCSCMatrix(chrom, ss.csc_matrix([], shape=(0, 0), dtype=data.dtype))
        m._data = data
        m._indices = indices
        m._indptr = indptr
        m._shape = shape
        m._matrix_type_str = "CSC"
        m._matrix_type = ss.csc_matrix

        # Assertion commented out for performance reasons
        # assert ss.triu(m.get(), k=1).sum() == 0 or ss.tril(m.get(), k=-1).sum() == 0

        return m

    @property
    def T(self):  # noqa
        return SharedTriangularCSRMatrix._from_shared_buffers(  # noqa
            self._chrom,
            self._data,
            self._indices,
            self._indptr,
            self._shape,
        )


class SharedTriangularSparseMatrix(object):
    def __init__(self, chrom: str, m: SparseMatrix, logger=None, max_nnz: Optional[int] = None):
        if isinstance(m, ss.csr_matrix):
            self._m = SharedTriangularCSRMatrix(chrom, m, logger, max_nnz)
        elif isinstance(m, ss.csc_matrix):
            self._m = SharedTriangularCSCMatrix(chrom, m, logger, max_nnz)
        else:
            self._m = SharedTriangularCSCMatrix(chrom, ss.csc_matrix(m), logger, max_nnz)

    def get(self) -> SparseMatrix:
        return self._m.get()

    def can_assign(self, m) -> bool:
        if isinstance(m, ss.csr_matrix):
            return isinstance(self._m, SharedTriangularCSRMatrix) and self._m.can_assign(m)
        if isinstance(m, ss.csc_matrix):
            return isinstance(self._m, SharedTriangularCSCMatrix) and self._m.can_assign(m)

        raise NotImplementedError

    def assign(self, chrom: str, m: SparseMatrix, logger=None):
        assert self.can_assign(m) and self._m.can_assign(m)
        self._m.assign(chrom, m, logger)

    def update_metadata(self, chrom, shape, data_shape, indices_shape, indptr_shape, sum):
        self._m.update_metadata(chrom, shape, data_shape, indices_shape, indptr_shape, sum)

    @property
    def metadata(self) -> Dict:
        return self._m.metadata

    @property
    def T(self) -> Union[SharedTriangularCSRMatrix, SharedTriangularCSCMatrix]:
        return self._m.T


def set_shared_state(lt_matrix: SharedTriangularSparseMatrix, ut_matrix: SharedTriangularSparseMatrix):
    if lt_matrix is not None:
        global _lower_triangular_matrix
        _lower_triangular_matrix = lt_matrix
    if ut_matrix is not None:
        global _upper_triangular_matrix
        _upper_triangular_matrix = ut_matrix


def unset_shared_state():
    call_gc = False
    if shared_state_avail("lower"):
        global _lower_triangular_matrix
        del _lower_triangular_matrix
        call_gc = True

    if shared_state_avail("upper"):
        global _upper_triangular_matrix
        del _upper_triangular_matrix
        call_gc = True

    if call_gc:
        gc.collect()


def shared_state_avail(key: str) -> bool:
    if key in {"lower", "LT"}:
        return "_lower_triangular_matrix" in globals()

    if key in {"upper", "UT"}:
        return "_upper_triangular_matrix" in globals()

    raise ValueError("Invalid key")


def get_shared_state(key: str, metadata: Optional[Dict] = None) -> SharedTriangularSparseMatrix:
    if key in {"lower", "LT"}:
        global _lower_triangular_matrix
        if metadata is not None and _lower_triangular_matrix.metadata != metadata:
            _lower_triangular_matrix.update_metadata(**metadata)
        return _lower_triangular_matrix

    if key in {"upper", "UT"}:
        global _upper_triangular_matrix
        if metadata is not None and _upper_triangular_matrix.metadata != metadata:
            _upper_triangular_matrix.update_metadata(**metadata)
        return _upper_triangular_matrix
    raise ValueError("Invalid key")
