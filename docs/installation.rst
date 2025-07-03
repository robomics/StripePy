..
  Copyright (C) 2025 Andrea Raffo <andrea.raffo@ibv.uio.no>
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

Installing from source requires git to be available on the host.

Installing the latest version from the main branch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  pip install 'stripepy-hic[all] @ git+https://github.com/paulsengroup/StripePy.git@main'

Installing version corresponding to a git tag
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  pip install 'stripepy-hic[all] @ git+https://github.com/paulsengroup/StripePy.git@v1.1.0'

Installing from a release archive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  pip install 'stripepy-hic[all] @ https://pypi.python.org/packages/source/s/stripepy_hic/stripepy_hic-1.1.0.tar.gz'
