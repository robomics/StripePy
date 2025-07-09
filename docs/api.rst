..
   Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
   SPDX-License-Identifier: MIT

Python API Reference
####################

.. py:module:: stripepy
.. py:currentmodule:: stripepy.data_structures

.. autodata:: SparseMatrix

.. autoclass:: Stripe

  .. automethod:: __init__
  .. autoproperty:: seed
  .. autoproperty:: top_persistence
  .. autoproperty:: lower_triangular
  .. autoproperty:: upper_triangular
  .. autoproperty:: left_bound
  .. autoproperty:: right_bound
  .. autoproperty:: top_bound
  .. autoproperty:: bottom_bound
  .. autoproperty:: inner_mean
  .. autoproperty:: inner_std
  .. autoproperty:: five_number
  .. autoproperty:: outer_lsum
  .. autoproperty:: outer_rsum
  .. autoproperty:: outer_lsize
  .. autoproperty:: outer_rsize
  .. autoproperty:: outer_lmean
  .. autoproperty:: outer_rmean
  .. autoproperty:: outer_mean
  .. autoproperty:: rel_change
  .. automethod:: set_horizontal_bounds
  .. automethod:: set_vertical_bounds
  .. automethod:: compute_biodescriptors

.. autoclass:: ResultFile

  .. automethod:: __init__
  .. automethod:: create
  .. automethod:: create_from_file
  .. automethod:: append
  .. autoproperty:: assembly
  .. autoproperty:: chromosomes
  .. autoproperty:: creation_date
  .. autoproperty:: format
  .. autoproperty:: format_url
  .. autoproperty:: format_version
  .. autoproperty:: generated_by
  .. autoproperty:: metadata
  .. autoproperty:: normalization
  .. autoproperty:: path
  .. autoproperty:: resolution
  .. automethod:: finalize
  .. automethod:: __getitem__
  .. automethod:: get_min_persistence
  .. automethod:: get
  .. automethod:: write_descriptors


.. autoclass:: Result

  .. automethod:: __init__
  .. autoproperty:: chrom
  .. autoproperty:: empty
  .. autoproperty:: min_persistence
  .. autoproperty:: roi
  .. automethod:: get(name: str, location: str) -> List[Stripe] | numpy.ndarray[int] | numpy.ndarray[float]
  .. automethod:: get_stripes_descriptor(descriptor: str, location: str) -> numpy.ndarray[int] | numpy.ndarray[float]
  .. automethod:: get_stripe_bio_descriptors
  .. automethod:: get_stripe_geo_descriptors
