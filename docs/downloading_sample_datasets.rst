..
  Copyright (C) 2025 Andrea Raffo <andrea.raffo@ibv.uio.no>
  SPDX-License-Identifier: MIT

Downloading sample datasets
===========================

`stripepy_download_help` is used to download sample datasets suitable to quickly test StripePy.
It provides various options to specify which datasets to download, where to store them, and how to handle existing files.

Listing available datasets
--------------------------

Before downloading, you might want to see the available datasets.
The ``--list-only`` options prints a nested JSON object without initiating any downloads.

The outermost JSON object has dataset names as keys.
Each value is an object containing metadata like URL, assembly, and file format.

.. code-block:: console

  user@dev:/tmp$ stripepy download --list-only

  {
    "4DNFI3RFZLZ5": {
      "url": "https://zenodo.org/records/15301784/files/4DNFI3RFZLZ5.stripepy.mcool?download=1",
      "md5": "f6e060211c95dd5fbf6e708c637d1c1c",
      "assembly": "mm10",
      "format": "mcool",
      "size_mb": 83.85
    },
    ...
    "ENCFF993FGR": {
      "url": "https://zenodo.org/records/15301784/files/ENCFF993FGR.stripepy.hic?download=1",
      "md5": "3bcb8c8c5aac237f26f994e0f5e983d7",
      "assembly": "hg38",
      "format": "hic",
      "size_mb": 185.29
    }
  }


Basic download operations
-------------------------

`stripepy_download_help` can be used to download a dataset by specifying its name.

For example, to download the dataset named ``4DNFI9GMP2J8``:

.. code-block:: console

  # This may take a while on slow internet connections
  user@dev:/tmp$ stripepy download --name 4DNFI9GMP2J8

  2025-01-14 12:46:01.304277 [info     ] downloading dataset "4DNFI9GMP2J8" (assembly=hg38)...
  2025-01-14 12:46:23.900411 [info     ] DONE! Downloading dataset "4DNFI9GMP2J8" took 22.596s.
  2025-01-14 12:46:23.901141 [info     ] computing MD5 digest for file "/tmp/4DNFI9GMP2J8.dvizz7v1"...
  2025-01-14 12:46:24.050566 [info     ] MD5 checksum match!
  2025-01-14 12:46:24.050695 [info     ] successfully downloaded dataset "https://zenodo.org/records/14643417/files/4DNFI9GMP2J8.stripepy.mcool?download=1" to file "4DNFI9GMP2J8.mcool"
  2025-01-14 12:46:24.050752 [info     ] file size: 106.84MB. Elapsed time: 22.979s

Note that, if the dataset already exists, it will be overwritten if and only if the ``--force`` flag is specified; otherwise, the command will fail to prevent accidental data loss.

If no name is provided, the tool will default to selecting a random dataset.
You can refine this random selection by specifying parameters like the maximum allowed size:

.. code-block:: console

  # This may take a while on slow internet connections
  user@dev:/tmp$ stripepy download --max-size 100

  2025-07-02 12:29:43.095292 [info     ] downloading dataset "4DNFI3RFZLZ5" (assembly=mm10)...
  2025-07-02 12:30:31.664521 [info     ] DONE! Downloading dataset "4DNFI3RFZLZ5" took 48.569s.
  2025-07-02 12:30:31.665930 [info     ] computing MD5 digest for file "/tmp/4DNFI3RFZLZ5._3a7e1bs"...
  2025-07-02 12:30:31.850671 [info     ] MD5 checksum match!
  2025-07-02 12:30:31.851358 [info     ] successfully downloaded dataset "https://zenodo.org/records/15301784/files/4DNFI3RFZLZ5.stripepy.mcool?download=1" to file "4DNFI3RFZLZ5.mcool"
  2025-07-02 12:30:31.851449 [info     ] file size: 83.86 MiB. Elapsed time: 49.160s

The ``--assembly`` option provides a mechanism to restrict downloads to datasets specifically associated with a given reference genome assembly.
Currently, the supported options for this parameter are ``hg38`` (human) and ``mm10`` (mouse).

Additionally, for internal testing or specific development purposes, the ``--include-private`` flag allows the inclusion of datasets typically for internal testing.

Downloading test datasets
-------------------------

The `stripepy_download_help` command also provides dedicated functionalities for acquiring specific test datasets used in unit and end-to-end testing.

The ``--unit-test`` option allows the download of datasets specifically required by the unit tests.
When this option is invoked, the downloaded files are automatically stored within the ``test/data/`` directory, and any existing files at that location will be overwritten.
It is important to note that when ``--unit-test`` is specified, all other command-line options are ignored, streamlining the test data acquisition process.

Similarly, the ``--end2end-test`` option allows the download of datasets necessary for end-to-end tests.
Consistent with the unit test behavior, these files are also stored in the ``test/data/`` directory, existing files are overwritten, and all other command-line options are disregarded when this flag is active.
