<!--
Copyright (C) 2024 Roberto Rossini <roberros@uio.no>

SPDX-License-Identifier: MIT
-->

# StripePy test instructions

The instructions in this README assume that StripePy and pytest have been installed in a virtual environment named `venv`.

The provided commands should work on any UNIX system. To run the test suites on Windows simply replace `venv/bin/` with `venv\Scripts\`.

### Downloading test files

```console
user@dev:/tmp/StripePy$ mkdir -p test/data/

user@dev:/tmp/StripePy$ venv/bin/stripepy download --name 4DNFI9GMP2J8 -o test/data/4DNFI9GMP2J8.mcool
[2024-12-05 19:38:45,416] INFO: downloading dataset "4DNFI9GMP2J8" (assembly=hg38)...
[2024-12-05 19:38:45,655] INFO: downloaded 0.00/106.84 MB (0.00%)
[2024-12-05 19:39:00,669] INFO: downloaded 49.06/106.84 MB (45.92%)
[2024-12-05 19:39:15,676] INFO: downloaded 102.91/106.84 MB (96.33%)
[2024-12-05 19:39:16,676] INFO: DONE! Downloading dataset "4DNFI9GMP2J8" took 31.26s.
[2024-12-05 19:39:16,677] INFO: computing MD5 digest for file "/tmp/StripePy/test/data/4DNFI9GMP2J8.om0wf3sr"...
[2024-12-05 19:39:16,830] INFO: MD5 checksum match!
[2024-12-05 19:39:16,830] INFO: successfully downloaded dataset "https://zenodo.org/records/14283922/files/4DNFI9GMP2J8.stripepy.mcool?download=1" to file "test/data/4DNFI9GMP2J8.mcool"
[2024-12-05 19:39:16,830] INFO: file size: 106.84MB. Elapsed time: 31.41s

user@dev:/tmp/StripePy$ venv/bin/stripepy download --name __results_v1 -o test/data/results_4DNFI9GMP2J8_v1.hdf5
[2024-12-05 19:42:28,838] INFO: downloading dataset "__results_v1" (assembly=hg38)...
[2024-12-05 19:42:29,088] INFO: downloaded 0.00/8.75 MB (0.00%)
[2024-12-05 19:42:30,176] INFO: DONE! Downloading dataset "__results_v1" took 1.34s.
[2024-12-05 19:42:30,176] INFO: computing MD5 digest for file "/tmp/StripePy/test/data/results_4DNFI9GMP2J8_v1.lrex4ftr"...
[2024-12-05 19:42:30,189] INFO: MD5 checksum match!
[2024-12-05 19:42:30,189] INFO: successfully downloaded dataset "https://zenodo.org/records/14283922/files/results_4DNFI9GMP2J8_v1.hdf5?download=1" to file "test/data/results_4DNFI9GMP2J8_v1.hdf5"
[2024-12-05 19:42:30,189] INFO: file size: 8.75MB. Elapsed time: 1.35s

user@dev:/tmp/StripePy$ venv/bin/stripepy download --name "__stripepy_plot_images" -o test/data/stripepy-plot-test-images.tar.xz
[2024-12-18 14:40:00,616] INFO: downloading dataset "__stripepy_plot_images" (assembly=hg38)...
[2024-12-18 14:40:00,788] INFO: downloaded 0.00/1.49 MB (0.00%)
[2024-12-18 14:40:01,073] INFO: DONE! Downloading dataset "__stripepy_plot_images" took 0.46s.
[2024-12-18 14:40:01,073] INFO: computing MD5 digest for file "test/data/stripepy-plot-test-images.tar.54puv9p_"...
[2024-12-18 14:40:01,075] INFO: MD5 checksum match!
[2024-12-18 14:40:01,075] INFO: successfully downloaded dataset "https://zenodo.org/records/14510452/files/stripepy-plot-test-images.tar.xz?download=1" to file "test/data/stripepy-plot-test-images.tar.xz"
[2024-12-18 14:40:01,075] INFO: file size: 1.49MB. Elapsed time: 0.46s
```

## Running the unit tests

```console
user@dev:/tmp/StripePy$ venv/bin/pytest test/ -v -m unit

============================================= test session starts =============================================
platform linux -- Python 3.12.7, pytest-8.3.4, pluggy-1.5.0 -- /tmp/StripePy/venv/bin/python
cachedir: .pytest_cache
rootdir: /tmp/StripePy
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

<!--
# To re-center the header/footer message

```python
x = " 5 passed, 11 deselected in 55.71s "
x.center(113, "=")
```
-->

```console
user@dev:/tmp/StripePy$ venv/bin/pytest test/ -v -m end2end

============================================== test session starts ==============================================
platform linux -- Python 3.13.0, pytest-8.3.4, pluggy-1.5.0 -- /home/roberros/github/StripePy/venv/bin/python3
cachedir: .pytest_cache
rootdir: /home/roberros/github/StripePy
configfile: pyproject.toml
testpaths: test/
plugins: cov-6.0.0
collected 31 items / 19 deselected / 12 selected

test/integration/test_stripepy_call.py::TestStripePyCall::test_stripepy_call PASSED                                                     [  8%]
test/integration/test_stripepy_download.py::TestStripePyDownload::test_list_only PASSED                                                 [ 16%]
test/integration/test_stripepy_download.py::TestStripePyDownload::test_download_by_name PASSED                                          [ 25%]
test/integration/test_stripepy_download.py::TestStripePyDownload::test_download_random PASSED                                           [ 33%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_contact_map PASSED                                                       [ 41%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_contact_map_with_seeds PASSED                                            [ 50%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_contact_map_with_stripes PASSED                                          [ 58%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_contact_map_with_stripes_no_heights PASSED                               [ 66%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_pseudodistribution PASSED                                                [ 75%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_stripe_hist PASSED                                                       [ 83%]
test/integration/test_stripepy_plot.py::TestStripePyPlot::test_stripe_hist_gw PASSED                                                    [ 91%]
test/integration/test_stripepy_view.py::TestStripePyView::test_view PASSED                                                              [100%]

================================== 12 passed, 19 deselected in 80.99s (0:01:20) =================================
```
