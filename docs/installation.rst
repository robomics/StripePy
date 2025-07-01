..
   Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
   SPDX-License-Identifier: MIT

Installation
============

StripePy can be installed in various ways.

Installing with pip
-------------------

.. code-block:: bash

  pip install 'stripepy-hic[all]'


Installing with conda
---------------------

.. code-block:: bash

  conda create -n stripepy -c conda-forge -c bioconda stripepy-hic

Installing from source
----------------------

Instructions for Linux and macOS:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  # create and activate a venv (optional)
  python3 -m venv venv
  . venv/bin/activate

  # get StripePy source code
  git clone https://github.com/paulsengroup/StripePy.git

  # optional, checkout a specific version
  # git checkout v0.0.2

  # install StripePy
  cd StripePy
  pip install '.[all]'

  # ensure StripePy is in your PATH
  stripepy --help

Instructions for Windows:
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  # create and activate a venv (optional)
  python3 -m venv venv
  venv\Scripts\activate

  # get StripePy source code
  git clone https://github.com/paulsengroup/StripePy.git

  # optional, checkout a specific version
  # git checkout v0.0.2

  # install StripePy
  cd StripePy
  pip install .

  # ensure StripePy is in your PATH
  stripepy --help
