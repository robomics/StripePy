<!--
Copyright (C) 2024 Roberto Rossini <roberros@uio.no>

SPDX-License-Identifier: MIT
-->

# StripePy test instructions

The instructions in this README assume that StripePy and pytest have been installed in a virtual environment named `venv`.

The provided commands should work on any UNIX system. To run the test suites on Windows simply replace `venv/bin/` with `venv\Scripts\`.

## Running the unit tests

```console
user@dev:/tmp$ venv/bin/pytest test/ -v -m unit

============================================= test session starts =============================================
platform linux -- Python 3.12.7, pytest-8.3.4, pluggy-1.5.0 -- /tmp/venv/bin/python
cachedir: .pytest_cache
rootdir: /tmp
configfile: pyproject.toml
plugins: anyio-4.6.2.post1, cov-6.0.0
collected 15 items / 6 deselected / 9 selected

test/unit/test_IO.py::TestRemoveAndCreateFolder::test_create_new_folder PASSED                           [ 11%]
test/unit/test_IO.py::TestRemoveAndCreateFolder::test_overwrite_existing_folder PASSED                   [ 22%]
test/unit/test_others.py::TestCmapLoading::test_invalid_paths PASSED                                     [ 33%]
test/unit/test_others.py::TestCmapLoading::test_invalid_formats PASSED                                   [ 44%]
test/unit/test_others.py::TestCmapLoading::test_invalid_resolutions PASSED                               [ 55%]
test/unit/test_others.py::TestCmapLoading::test_valid_files PASSED                                       [ 66%]
test/unit/test_stripepy.py::TestLogTransform::test_empty PASSED                                          [ 77%]
test/unit/test_stripepy.py::TestLogTransform::test_all_finite PASSED                                     [ 88%]
test/unit/test_stripepy.py::TestLogTransform::test_with_nans PASSED                                      [100%]

======================================= 9 passed, 6 deselected in 0.92s =======================================
```

## Running the integration tests

### Downloading test files

TODO

### Running the test suite

```console
user@dev:/tmp$ venv/bin/pytest test/ -v -m end2end
============================================= test session starts =============================================
platform linux -- Python 3.12.7, pytest-8.3.4, pluggy-1.5.0 -- /tmp/venv/bin/python
cachedir: .pytest_cache
rootdir: /tmp
configfile: pyproject.toml
plugins: anyio-4.6.2.post1, cov-6.0.0
collected 15 items / 11 deselected / 4 selected

test/integration/test_stripepy_call.py::TestStripePyCall::test_stripepy_call PASSED                      [ 25%]
test/integration/test_stripepy_download.py::TestStripePyDownload::test_list_only PASSED                  [ 50%]
test/integration/test_stripepy_download.py::TestStripePyDownload::test_download_by_name PASSED           [ 75%]
test/integration/test_stripepy_download.py::TestStripePyDownload::test_download_random PASSED            [100%]

======================================= 4 passed, 11 deselected in 6.97s ======================================
```
