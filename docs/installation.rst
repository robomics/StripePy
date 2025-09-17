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

  pip install 'stripepy-hic[all] @ git+https://github.com/paulsengroup/StripePy.git@v1.2.0'

Installing from a release archive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

  pip install 'stripepy-hic[all] @ https://pypi.python.org/packages/source/s/stripepy_hic/stripepy_hic-1.2.0.tar.gz'


Containers (Docker or Singularity/Apptainer)
--------------------------------------------

First, ensure you have followed the instructions on how to install Docker or Singularity/Apptainer on your OS.

.. raw:: html

   <details>
   <summary><a>Installing Docker</a></summary>

The following instructions assume you have root/admin permissions.

* `Linux <https://docs.docker.com/desktop/install/linux-install/>`_
* `macOS <https://docs.docker.com/desktop/install/mac-install/>`_
* `Windows <https://docs.docker.com/desktop/install/windows-install/>`_

On some Linux distributions, simply installing Docker is not enough.
You also need to start (and optionally enable) the appropriate service(s).
This is usually done with one of the following:

.. code-block:: sh

  sudo systemctl start docker
  sudo systemctl start docker.service


Refer to `Docker <https://docs.docker.com/engine/install/>`_ or your OS/distribution documentation for more details.

.. raw:: html

   </details>

Pulling stripepy Docker image
-----------------------------

stripepy Docker images are available on `GHCR.io <https://github.com/paulsengroup/stripepy/pkgs/container/stripepy>`_
and `DockerHub <https://hub.docker.com/r/paulsengroup/stripepy>`_.

Downloading and running the latest stable release can be done as follows:

.. code-block:: console

  # Using Docker, may require sudo
  user@dev:/tmp$ docker run ghcr.io/paulsengroup/stripepy:1.2.0 --help

  # Using Singularity/Apptainer
  user@dev:/tmp$ singularity run ghcr.io/paulsengroup/stripepy:1.2.0 --help

  usage: stripepy [-h] [-v] {call,download,plot,view} ...

  stripepy is designed to recognize linear patterns in contact maps (.hic, .mcool, .cool) through the geometric reasoning, including topological persistence and quasi-interpolation.

  options:
    -h, --help            show this help message and exit
    -v, --version         show program's version number and exit

  subcommands:
    {call,download,plot,view}
                          List of available subcommands:
      call                stripepy works in four consecutive steps:
                          • Step 1: Pre-processing
                          • Step 2: Recognition of loci of interest (also called 'seeds')
                          • Step 3: Shape analysis (i.e., width and height estimation)
                          • Step 4: Signal analysis
      download            Helper command to simplify downloading datasets that can be used to test StripePy.
      plot                Generate various static plots useful to visually inspect the output produced by stripepy call.
      view                Fetch stripes from the HDF5 file produced by stripepy call.

The above will print stripepy's help message, and is equivalent to running :code:`stripepy --help` from the command line (assuming stripepy is available on your machine).
