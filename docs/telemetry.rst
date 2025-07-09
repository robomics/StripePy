..
  Copyright (C) 2025 Andrea Raffo <andrea.raffo@ibv.uio.no>
  SPDX-License-Identifier: MIT

Telemetry
#########

Starting with version v1.1.1 of StripePy, we introduced support for telemetry collection.

This page outlines what information we are collecting and why.
Furthermore, we provide instructions on how telemetry collection can be disabled.

What information is being collected
-----------------------------------

``stripepy`` is instrumented to collect general information about ``stripepy`` itself and the system where it is being run.

We do not collect any sensitive information that could be used to identify our users, the machine where ``stripepy`` is being run, or the datasets processed by ``stripepy``.

This is the data we are collecting:

* Information on how ``stripepy`` was installed (i.e., the package version and third-party dependency versions).
* Information on the system where ``stripepy`` is being run (i.e., operating system, processor architecture, and Python version).
* How ``stripepy`` is being invoked (i.e., the subcommand, input file format and parameters).
* Information about ``stripepy`` execution (i.e., when it was launched, how long the command took to finish, and whether the command terminated with an error).

The following table shows an example of the telemetry collected when running ``stripepy call ENCFF993FGR.hic 10000 -p 8``:

.. csv-table:: Telemetry information collected when running ``stripepy call``
  :file: ./assets/telemetry_table.tsv
  :header-rows: 1
  :delim: tab

Why are we collecting this information?
---------------------------------------

There are two main motivations behind our decision to start collecting telemetry data:

#. To get an idea of how big our user base is: this will help us, among other things, to secure funding to maintain ``stripepy`` in the future.
#. To better understand which of the functionalities offered by ``stripepy`` are most used by our users: we intend to use this information to help us decide which features we should focus our development efforts on.

How is telemetry information processed and stored
-------------------------------------------------

Telemetry is sent to an OpenTelemetry collector running on a virtual server hosted on the Norwegian Research and Education Cloud (`NREC <https://www.nrec.no/>`_).

The virtual server and collector are managed by us, and traffic between ``stripepy`` and the collector is encrypted.

The collector processes incoming data continuously and forwards it to a dashboard for data analytics and a backup solution (both services are hosted in Europe).
Communication between the collector, dashboard, and backup site is also encrypted.
Data stored by the dashboard and backup site is encrypted at rest.

The analytics dashboard keeps telemetry data for up to 60 days, while the backup site is currently set up to store telemetry data indefinitely (although this may change in the future).

How to disable telemetry collection
-----------------------------------

To disable telemetry collection, simply define the ``STRIPEPY_NO_TELEMETRY`` environment variable before launching ``stripepy`` (e.g., ``STRIPEPY_NO_TELEMETRY=1 stripepy download``)

Where can I find the code used for telemetry collection?
--------------------------------------------------------

All code concerning telemetry collection is defined in file `src/stripepy/cli/telemetry.py <https://github.com/paulsengroup/StripePy/blob/main/src/stripepy/cli/telemetry.py>`_.
