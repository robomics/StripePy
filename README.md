<!--
Copyright (C) 2024 Roberto Rossini <roberros@uio.no>

SPDX-License-Identifier: MIT
-->

# StripePy

---

<!-- markdownlint-disable MD033 -->

<table>
    <tr>
      <td>Paper</td>
      <td>
        <a href="https://doi.org/10.1093/bioinformatics/btaf351">
          <img src="https://img.shields.io/badge/CITE-Bioinformatics%20(2025)-blue" alt="Bioinformatics 2025">
        </a>
      </td>
    </tr>
    <tr>
      <td>Downloads</td>
      <td>
        <a href="https://anaconda.org/bioconda/stripepy-hic">
          <img src="https://img.shields.io/conda/vn/bioconda/stripepy-hic?label=bioconda&logo=Anaconda" alt="Bioconda">
        </a>
        &nbsp
        <a href="https://pypi.org/project/stripepy-hic/">
          <img src="https://img.shields.io/pypi/v/stripepy-hic" alt="PyPI">
        </a>
        &nbsp
        <a href="https://doi.org/10.5281/zenodo.14394041">
          <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.14394042.svg" alt="Zenodo">
        </a>
      </td>
    </tr>
    <tr>
    <tr>
        <td>Documentation</td>
        <td>
          <a href="https://stripepy.readthedocs.io">
            <img src="https://app.readthedocs.org/projects/stripepy/badge/?version=stable&style=flat" alt="Documentation">
          </a>
        </td>
      </tr>
      <td>CI</td>
      <td>
        <a href="https://github.com/paulsengroup/StripePy/actions/workflows/ci.yml">
          <img src="https://github.com/paulsengroup/StripePy/actions/workflows/ci.yml/badge.svg" alt="Ubuntu CI Status">
        </a>
        &nbsp
        <a href="https://github.com/paulsengroup/StripePy/actions/workflows/build-dockerfile.yml">
          <img src="https://github.com/paulsengroup/StripePy/actions/workflows/build-dockerfile.yml/badge.svg" alt="Build Dockerfile Status">
        </a>
      </td>
    </tr>
    <tr>
        <td>License</td>
        <td>
          <a href="https://github.com/paulsengroup/StripePy/blob/main/LICENCE">
            <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
          </a>
        </td>
      </tr>
</table>

<!-- markdownlint-enable MD033 -->

---

StripePy is a CLI application written in Python that recognizes architectural stripes found in the interaction matrix files
generated by Chromosome Conformation Capture experiments, such as Hi-C and Micro-C.

StripePy is developed on Linux and macOS and is also tested on Windows. Installing StripePy is quick and easy using pip:

```bash
pip install 'stripepy-hic[all]'
```

For other installation options (conda, source, and Docker or Singularity/Apptainer), and details on ensuring StripePy is in your `PATH`, please refer to the official [documentation](https://stripepy.readthedocs.io).

## Why Choose StripePy?

StripePy stands out with several key features that make it a fast and robust stripe caller:

- **Broad Format Support**: Compatible with major formats: `.hic`, `.cool` and `.mcool`; outputs to `.hdf5` and `BEDPE`.
- **User-Friendly**: Designed with an intuitive command-line interface, making stripe analysis accessible even to less experienced users.
- **Stripe descriptors**: Computes stripe width, height, and generates various statistics for post-processing, e.g., ranking and filtering.
- **Optimized performance**: Outperforms other tools over diverse datasets and a simulated benchmark, StripeBench.
- **Exceptional speed & Low Memory**: Significantly faster than existing tools (2x Chromosight, 66x Stripenn), with much lower memory usage.

## Key Features

StripePy is organized into a few subcommands:

<!-- markdownlint-disable MD059 -->

- `stripepy download`: download a minified sample dataset suitable to quickly test StripePy - [link](https://stripepy.readthedocs.io/en/stable/downloading_sample_datasets.html).
- `stripepy call`: run the stripe detection algorithm and store the identified stripes in a `.hdf5` file - [link](https://stripepy.readthedocs.io/en/stable/detect_stripes.html).
- `stripepy view`: take the `result.hdf5` file generated by `stripepy call` and extract stripes in BEDPE format - [link](https://stripepy.readthedocs.io/en/stable/fetch_stripes.html).
- `stripepy plot`: generate various kinds of plots to inspect the stripes identified by `stripepy call`- [link](https://stripepy.readthedocs.io/en/stable/generate_plots.html).

<!-- markdownlint-enable MD059 -->

For a quick introduction to the tool, refer to the [Quickstart](https://stripepy.readthedocs.io/en/stable/quickstart.html) section in the documentation.

![Graphical Abstract](https://github.com/paulsengroup/StripePy/blob/75e87126058c7c825d87abbead717ceae7eeb8f2/docs/assets/pipeline-short.jpeg?raw=true)

For more information on the subcommands, please run `stripepy --help` and refer to the [documentation](https://stripepy.readthedocs.io/en/stable/cli_reference.html) and the [paper](https://doi.org/10.1093/bioinformatics/btaf351).

## Getting help

For any issues regarding StripePy installation, walkthrough, and output interpretation please open a [discussion](https://github.com/paulsengroup/StripePy/discussions) on GitHub.

If you've found a bug or would like to suggest a new feature, please open a new [issue](https://github.com/paulsengroup/StripePy/issues) instead.

## Citing

If you use StripePy in your research, please cite the following publication:

Andrea Raffo, Roberto Rossini, Jonas Paulsen\
StripePy: fast and robust characterization of architectural stripes\
_Bioinformatics_, Volume 41, Issue 6, June 2025, btaf351\
[https://doi.org/10.1093/bioinformatics/btaf351](https://doi.org/10.1093/bioinformatics/btaf351)

<details>
<summary>BibTex</summary>

```bibtex
@article{stripepy,
    author = {Raffo, Andrea and Rossini, Roberto and Paulsen, Jonas},
    title = {{StripePy: fast and robust characterization of architectural stripes}},
    journal = {Bioinformatics},
    volume = {41},
    number = {6},
    pages = {btaf351},
    year = {2025},
    month = {06},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btaf351},
    url = {https://doi.org/10.1093/bioinformatics/btaf351},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/41/6/btaf351/63484367/btaf351.pdf},
}
```

</details>
