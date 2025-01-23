# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import ctypes
import gc
import multiprocessing as mp
import time
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import scipy.sparse as ss

from .common import pretty_format_elapsed_time

SparseMatrix = Union[ss.csr_matrix, ss.csc_matrix]


class _SharedRawArrayWrapper(object):
    def __init__(self, data: Union[bytes, npt.NDArray], capacity: Optional[int] = None):
        if capacity is None:
            capacity = len(data)

        assert capacity >= len(data)

        self._dtype = bytes if isinstance(data, bytes) else data.dtype
        self._capacity = capacity
        self._size = mp.RawValue(ctypes.c_int, 0)
        if len(data) == 0:
            self._shared_buffer = None
        elif self._dtype == bytes:
            self._shared_buffer = mp.RawArray(ctypes.c_char, self._capacity)
        else:
            self._shared_buffer = mp.RawArray(np.ctypeslib.as_ctypes_type(self._dtype), self._capacity)

        self.assign(data)

    def __len__(self) -> int:
        return int(self._size.value)

    def can_assign(self, data: Union[bytes, npt.NDArray]) -> bool:
        if len(data) > self._capacity:
            return False

        if isinstance(data, bytes):
            return self._dtype == bytes

        return data.dtype == self._dtype

    def _assign_empty(self):
        self._size.value = 0

    def assign(self, data: Union[bytes, npt.NDArray]):
        assert self.can_assign(data)
        if len(data) == 0:
            self._assign_empty()
            return

        new_size = len(data)
        if self._dtype == bytes:
            self._shared_buffer[:new_size] = data
        else:
            dest = np.frombuffer(self._shared_buffer, dtype=self._dtype, count=new_size)
            np.copyto(dest, data, casting="safe")
        self._size.value = new_size

    def shrink(self, new_size: int):
        if new_size == 0:
            self._assign_empty()
            return

        assert new_size <= self._capacity
        self._size.value = new_size

    @property
    def dtype(self) -> Union[np.dtype, type]:
        return self._dtype

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def data(self) -> Union[bytes, npt.NDArray]:

        if self._dtype == bytes:
            return bytes(self._shared_buffer[: len(self)])
        return np.frombuffer(self._shared_buffer, dtype=self._dtype, count=len(self))

    @property
    def raw_data(self) -> mp.RawArray:
        return self._shared_buffer


class _SharedTriangularSparseMatrixBase(object):
    def __init__(self, chrom: Optional[str], m: Optional[SparseMatrix], logger=None, max_nnz: Optional[int] = None):
        assert isinstance(m, ss.csr_matrix) or isinstance(m, ss.csc_matrix) or m is None

        if chrom is None:
            # chrom should be None only if the object is to be initialized by a static function
            assert m is None
            assert max_nnz is None
            self._chrom = None
            self._data = None
            self._indices = None
            self._indptr = None
            self._shape = None
            self._matrix_type_str = None
            self._matrix_type = None
            return

        # hopefully nobody needs their chromosome names to be more than 1024 characters...
        capacity_chrom = max(1024, len(chrom))
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
        self._chrom = _SharedRawArrayWrapper(chrom.encode("ascii"), capacity_chrom)
        self._data = _SharedRawArrayWrapper(m.data, capacity_data)
        self._indices = _SharedRawArrayWrapper(m.indices, capacity_indices)
        self._indptr = _SharedRawArrayWrapper(m.indptr, capacity_indptr)
        self._shape = _SharedRawArrayWrapper(np.array(m.shape), 2)
        self._matrix_type_str = "CSR" if isinstance(m, ss.csr_matrix) else "CSC"
        self._matrix_type = type(m)

        if logger:
            logger.debug(
                "allocation and initialization of a %s matrix (%s; %d nnz) in shared memory took %s",
                self._matrix_type_str,
                chrom,
                m.nnz,
                pretty_format_elapsed_time(t0),
            )

    def get(self) -> SparseMatrix:
        return self._matrix_type(
            (self._data.data, self._indices.data, self._indptr.data),
            shape=self._shape.data,
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

        chrom_ascii = chrom.encode("ascii")
        if len(chrom_ascii) > self._chrom.capacity:
            # in case the chromosome name is too long, just truncate it
            chrom_ascii = chrom_ascii[: self._chrom.capacity]

        t0 = time.time()
        self._chrom.assign(chrom_ascii)
        self._data.assign(m.data)
        self._indices.assign(m.indices)
        self._indptr.assign(m.indptr)
        self._shape.assign(np.array(m.shape))

        if logger:
            logger.debug(
                "assigning to a %s matrix (%s; %d nnz) in shared memory took %s",
                self._matrix_type_str,
                chrom,
                m.nnz,
                pretty_format_elapsed_time(t0),
            )

    @property
    def chrom(self) -> str:
        return bytes(self._chrom.data).decode("ascii")

    @property
    def shape(self) -> npt.NDArray[int]:
        return self._shape.data


class SharedTriangularCSRMatrix(_SharedTriangularSparseMatrixBase):
    def __init__(self, chrom: str, m: ss.csr_matrix, logger=None, max_nnz: Optional[int] = None):
        assert isinstance(m, ss.csr_matrix) or m is None
        super().__init__(chrom, m, logger, max_nnz)

    @staticmethod
    def _from_shared_buffers(
        chrom: _SharedRawArrayWrapper,
        data: _SharedRawArrayWrapper,
        indices: _SharedRawArrayWrapper,
        indptr: _SharedRawArrayWrapper,
        shape: _SharedRawArrayWrapper,
    ):
        assert shape.data[0] == shape.data[1]
        m = SharedTriangularCSRMatrix(None, None, None, None)  # noqa
        m._chrom = chrom
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
        assert isinstance(m, ss.csc_matrix) or m is None
        super().__init__(chrom, m, logger, max_nnz)

    @staticmethod
    def _from_shared_buffers(
        chrom: _SharedRawArrayWrapper,
        data: _SharedRawArrayWrapper,
        indices: _SharedRawArrayWrapper,
        indptr: _SharedRawArrayWrapper,
        shape: _SharedRawArrayWrapper,
    ):
        assert shape.data[0] == shape.data[1]
        m = SharedTriangularCSCMatrix(None, None, None, None)  # noqa
        m._chrom = chrom
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

    @staticmethod
    def _empty(format: str):
        if format == "csr":
            return SharedTriangularSparseMatrix("", ss.csr_matrix([], shape=(0, 0), dtype=float))
        if format == "csc":
            return SharedTriangularSparseMatrix("", ss.csc_matrix([], shape=(0, 0), dtype=float))

        raise NotImplementedError

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

    @property
    def T(self):  # noqa
        if isinstance(self._m, SharedTriangularCSRMatrix):
            m = SharedTriangularSparseMatrix._empty("csc")
        else:
            m = SharedTriangularSparseMatrix._empty("csr")

        m._m = self._m.T
        return m


def set_shared_state(lt_matrix: SharedTriangularSparseMatrix, ut_matrix: SharedTriangularSparseMatrix):
    if lt_matrix is not None:
        global _lower_triangular_matrix  # noqa
        _lower_triangular_matrix = lt_matrix
    if ut_matrix is not None:
        global _upper_triangular_matrix  # noqa
        _upper_triangular_matrix = ut_matrix


def unset_shared_state():
    call_gc = False
    if shared_state_avail("lower"):
        global _lower_triangular_matrix  # noqa
        del _lower_triangular_matrix
        call_gc = True

    if shared_state_avail("upper"):
        global _upper_triangular_matrix  # noqa
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


def get_shared_state(key: str) -> SharedTriangularSparseMatrix:
    if key in {"lower", "LT"}:
        global _lower_triangular_matrix  # noqa
        return _lower_triangular_matrix

    if key in {"upper", "UT"}:
        global _upper_triangular_matrix  # noqa
        return _upper_triangular_matrix

    raise ValueError("Invalid key")
