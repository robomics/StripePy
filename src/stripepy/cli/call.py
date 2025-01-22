# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import concurrent.futures
import contextlib
import functools
import json
import multiprocessing as mp
import pathlib
import platform
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import hictkpy
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.sparse as ss
import structlog

from stripepy import IO, others, stripepy
from stripepy.cli import logging
from stripepy.utils.common import _import_matplotlib, pretty_format_elapsed_time
from stripepy.utils.multiprocess_sparse_matrix import (
    SharedSparseMatrix,
    set_shared_state,
    unset_shared_state,
)
from stripepy.utils.progress_bar import initialize_progress_bar


def _init_mpl_backend(skip: bool):
    if skip:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError:
        structlog.get_logger().warning("failed to initialize matplotlib backend")
        pass


def _init_shared_state(
    lower_triangular_matrix: Optional[SharedSparseMatrix],
    upper_triangular_matrix: Optional[SharedSparseMatrix],
    log_queue: mp.Queue,
    init_mpl: bool,
):
    logging.ProcessSafeLogger.setup_logger(log_queue)

    if lower_triangular_matrix is not None:
        assert upper_triangular_matrix is not None
        set_shared_state(lower_triangular_matrix, upper_triangular_matrix)

    _init_mpl_backend(skip=not init_mpl)


class ProcessPoolWrapper(object):
    def __init__(
        self,
        nproc: int,
        main_logger: logging.ProcessSafeLogger,
        lt_matrix: Union[ss.csc_matrix, ss.csr_matrix, None] = None,
        ut_matrix: Union[ss.csc_matrix, ss.csr_matrix, None] = None,
        init_mpl: bool = False,
        logger=None,
    ):
        self._pool = None
        self._lt_matrix = None
        self._ut_matrix = None

        if nproc > 1:
            if lt_matrix is not None:
                assert ut_matrix is not None
                self._lt_matrix = SharedSparseMatrix(lt_matrix, logger)
                self._ut_matrix = SharedSparseMatrix(ut_matrix, logger)
                set_shared_state(self._lt_matrix, self._ut_matrix)

            self._pool = concurrent.futures.ProcessPoolExecutor(  # noqa
                max_workers=nproc,
                initializer=_init_shared_state,
                initargs=(self._lt_matrix, self._ut_matrix, main_logger.log_queue, init_mpl),
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pool is None:
            return False

        if exc_type is not None:
            structlog.get_logger().debug("shutting down process pool due to the following exception: %s", exc_val)

        self._pool.shutdown(wait=True, cancel_futures=True)
        if self._lt_matrix is not None:
            unset_shared_state()
            self._lt_matrix = None
            self._ut_matrix = None

        self._pool = None
        return False

    @property
    def map(self, chunksize: int = 50):
        if self._pool is None:
            return map
        return functools.partial(self._pool.map, chunksize=chunksize)

    def submit(self, fx, *args, **kwargs) -> concurrent.futures.Future:
        if self._pool is None:
            fut = concurrent.futures.Future()
            try:
                fut.set_result(fx(*args, **kwargs))
            except Exception as e:
                fut.set_exception(e)
            return fut

        return self._pool.submit(fx, *args, **kwargs)

    @property
    def ready(self):
        return self._pool is not None


class IOManager(object):
    def __init__(
        self,
        matrix_path: pathlib.Path,
        result_path: pathlib.Path,
        resolution: int,
        normalization: Optional[str],
        genomic_belt: int,
        region_of_interest: Optional[str],
        nproc: int,
        metadata: Dict[str, Any],
        main_logger: logging.ProcessSafeLogger,
    ):
        self._path = matrix_path
        self._resolution = resolution
        self._normalization = "NONE" if normalization is None else normalization
        self._genomic_belt = genomic_belt
        self._roi = region_of_interest

        self._tpool = ProcessPoolWrapper(
            nproc,
            main_logger=main_logger,
            init_mpl=False,
        )
        self._tasks = {}

        logger = structlog.get_logger().bind(step="IO")
        logger.info('initializing result file "%s"...', result_path)
        with IO.ResultFile.create_from_file(
            result_path,
            mode="a",
            matrix_file=hictkpy.File(matrix_path, resolution),
            normalization=normalization,
            metadata=metadata,
        ) as h5:
            pass
        self._h5_path = result_path
        self._h5_pending_io_task = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._tpool.__exit__(exc_type, exc_val, exc_tb)

    @staticmethod
    def _fetch(
        path: pathlib.Path,
        resolution: int,
        normalization: str,
        genomic_belt: int,
        chrom_name: str,
        roi: Optional[Dict],
    ):
        t0 = time.time()
        logger = structlog.get_logger().bind(chrom=chrom_name, step="IO")

        logger.info("fetching interactions using normalization=%s", normalization)
        matrix = hictkpy.File(path, resolution=resolution).fetch(chrom_name, normalization=normalization).to_csr()
        logger.info("fetched %d pixels in %s", matrix.count_nonzero(), pretty_format_elapsed_time(t0))

        logger = structlog.get_logger().bind(chrom=chrom_name, step=(1,))
        logger.info("data pre-processing")
        t0 = time.time()
        lt_matrix, ut_matrix, roi_matrix = stripepy.step_1(
            matrix,
            genomic_belt,
            resolution,
            RoI=roi,
            logger=logger,
        )
        logger.info("preprocessing took %s", pretty_format_elapsed_time(t0))

        return lt_matrix, ut_matrix, roi_matrix

    def fetch_interaction_matrix(
        self, chrom_name: str, chrom_size: int
    ) -> Tuple[ss.csr_matrix, ss.csr_matrix, Optional[npt.NDArray]]:
        data = self.get_interaction_matrix(chrom_name)
        if data is not None:
            structlog.get_logger().bind(chrom=chrom_name, step="IO").info("returning pre-fetched interactions")
            return data

        roi = others.define_RoI(self._roi, chrom_size, self._resolution)
        return IOManager._fetch(
            self._path,
            self._resolution,
            self._normalization,
            self._genomic_belt,
            chrom_name,
            roi,
        )

    def fetch_interaction_matrix_async(self, chrom_name: str, chrom_size: int):
        if not self._tpool.ready:
            return

        assert chrom_name not in self._tasks

        roi = others.define_RoI(self._roi, chrom_size, self._resolution)
        self._tasks[chrom_name] = self._tpool.submit(
            IOManager._fetch,
            self._path,
            self._resolution,
            self._normalization,
            self._genomic_belt,
            chrom_name,
            roi,
        )

    def fetch_next_interaction_matrix_async(self, tasks: List[Tuple[str, int, bool]]):
        if not self._tpool.ready:
            return

        if len(tasks) == 0 or tasks[0][0] in self._tasks:
            return

        for chrom, size, skip in tasks:
            if not skip:
                self.fetch_interaction_matrix_async(chrom, size)
                return

    def get_interaction_matrix(
        self, chrom: str
    ) -> Optional[Tuple[ss.csr_matrix, ss.csr_matrix, Optional[npt.NDArray]]]:
        res = self._tasks.pop(chrom, None)
        if res is not None:
            res = res.result()
        return res

    @staticmethod
    def _write_results(path: pathlib.Path, result: IO.Result):
        logger = structlog.get_logger().bind(chrom=result.chrom[0], step="IO")
        logger.info('writing results to file "%s"', path)
        start_time = time.time()
        with IO.ResultFile.append(path) as h5:
            h5.write_descriptors(result)
        logger.info('successfully written results to "%s" in %s', path, pretty_format_elapsed_time(start_time))

    def _wait_on_io_on_results_file(self):
        """
        This checks for exceptions and ensures that we are not trying to perform concurrent writes on the same result file
        (e.g. starting to write results for chr2 before we're done writing results for chr1, or finalizing the result file
        before results for the last chromosome has been completely written to file)
        """

        if self._h5_pending_io_task is not None:
            self._h5_pending_io_task.result()
            self._h5_pending_io_task = None

    def write_results(self, result: IO.Result):
        self._wait_on_io_on_results_file()
        self._h5_pending_io_task = self._tpool.submit(IOManager._write_results, self._h5_path, result)

    def finalize_results(self):
        self._wait_on_io_on_results_file()
        with IO.ResultFile.append(self._h5_path) as h5:
            h5.finalize()


def _generate_metadata_attribute(
    constrain_heights: bool,
    genomic_belt: int,
    glob_pers_min: float,
    loc_pers_min: float,
    loc_trend_min: float,
    max_width: int,
    min_chrom_size: int,
) -> Dict[str, Any]:
    return {
        "constrain-heights": constrain_heights,
        "genomic-belt": genomic_belt,
        "global-persistence-minimum": glob_pers_min,
        "local-persistence-minimum": loc_pers_min,
        "local-trend-minimum": loc_trend_min,
        "max-width": max_width,
        "min-chromosome-size": min_chrom_size,
    }


def _plan(chromosomes: Dict[str, int], min_size: int, logger=None) -> List[Tuple[str, int, bool]]:
    plan = []
    small_chromosomes = []
    for chrom, length in chromosomes.items():
        skip = length <= min_size
        plan.append((chrom, length, skip))
        if skip:
            small_chromosomes.append(chrom)

    if len(small_chromosomes) != 0:
        if logger is None:
            logger = structlog.get_logger()

        logger.warning(
            "the following chromosomes are discarded because shorter than --min-chrom-size=%d bp: %s",
            min_size,
            ", ".join(small_chromosomes),
        )

    return plan


def _generate_empty_result(chrom: str, chrom_size: int, resolution: int) -> IO.Result:
    result = IO.Result(chrom, chrom_size)
    result.set_min_persistence(0)

    num_bins = (chrom_size + resolution - 1) // resolution
    for location in ("LT", "UT"):
        result.set("all_minimum_points", [], location)
        result.set("all_maximum_points", [], location)
        result.set("persistence_of_all_minimum_points", [], location)
        result.set("persistence_of_all_maximum_points", [], location)
        result.set("persistent_minimum_points", [], location)
        result.set("persistent_maximum_points", [], location)
        result.set("persistence_of_minimum_points", [], location)
        result.set("persistence_of_maximum_points", [], location)
        result.set("pseudodistribution", np.full(num_bins, np.nan, dtype=float), location)
        result.set("stripes", [], location)

    return result


class _JSONEncoder(json.JSONEncoder):
    def default(self, o: Any):
        if isinstance(o, pathlib.Path):
            return str(o)
        return super().default(o)


def _write_param_summary(config: Dict[str, Any], logger=None):
    if logger is None:
        logger = structlog.get_logger()

    config_str = json.dumps(config, indent=2, sort_keys=True, cls=_JSONEncoder)
    logger.info(f"CONFIG:\n{config_str}")


def _compute_progress_bar_weights(chrom_sizes: Dict[str, int], include_plotting: bool, nproc: int) -> pd.DataFrame:
    # These weights have been computed on a Linux machine (Ryzen 9 7950X3D) using 1 core to process
    # 4DNFI9GMP2J8.mcool at 10kbp
    step_weights = {
        "input": 0.035557,
        "step_1": 0.018159,
        "step_2": 0.015722,
        "step_3": 0.301879,
        "step_4": 0.051055,
        "output": 0.054285,
        "step_5": 0.523344 / nproc,
    }

    if not include_plotting:
        step_weights["step_5"] = 0.0

    tot = sum(step_weights.values())
    step_weights = {k: v / tot for k, v in step_weights.items()}

    weights = []
    for size in chrom_sizes.values():
        weights.extend((size * w for w in step_weights.values()))

    weights = np.array(weights)
    weights /= weights.sum()
    weights = np.minimum(weights.cumsum(), 1.0)

    shape = (len(chrom_sizes), len(step_weights))
    df = pd.DataFrame(weights.reshape(shape), columns=list(step_weights.keys()))
    df["chrom"] = list(chrom_sizes.keys())
    return df.set_index(["chrom"])


def _remove_existing_output_files(
    output_file: pathlib.Path, plot_dir: Optional[pathlib.Path], chromosomes: Dict[str, int]
):
    logger = structlog.get_logger()
    logger.debug("removing %s...", output_file)
    output_file.unlink(missing_ok=True)
    if plot_dir is not None:
        for path in plot_dir.glob("*"):
            if path.stem in chromosomes:
                logger.debug("removing %s...", path)
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


def _merge_results(futures) -> IO.Result:
    keys = (
        "all_minimum_points",
        "all_maximum_points",
        "persistence_of_all_minimum_points",
        "persistence_of_all_maximum_points",
        "persistent_minimum_points",
        "persistent_maximum_points",
        "persistence_of_minimum_points",
        "persistence_of_maximum_points",
        "pseudodistribution",
        "stripes",
    )

    results = {}
    i = 0
    for i, fut in enumerate(futures, 1):
        location, result = fut.result()
        results[location] = result

    assert len(results) == i

    result1, result2 = results["lower"], results["upper"]

    for key in keys:
        result1.set(key, result2.get(key, "upper"), "upper", force=True)

    return result1


def _setup_tpool(
    ctx,
    nproc: int,
    main_logger: logging.ProcessSafeLogger,
) -> Union[ProcessPoolWrapper, concurrent.futures.ThreadPoolExecutor]:
    if nproc > 1:
        return ctx.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=2))

    return ctx.enter_context(
        ProcessPoolWrapper(
            1,
            main_logger=main_logger,
            init_mpl=False,
        )
    )


def _run_step_2(
    chrom_name: str,
    chrom_size: int,
    ut_matrix: Optional[ss.csr_matrix],
    lt_matrix: Optional[ss.csr_matrix],
    min_persistence: float,
    tpool: concurrent.futures.ThreadPoolExecutor,
    pool: ProcessPoolWrapper,
    logger,
) -> IO.Result:
    logger = logger.bind(step=(2,))
    logger.info("topological data analysis")
    t0 = time.time()

    if platform.system() == "Linux":
        executor = pool
    else:
        executor = tpool

    task1 = executor.submit(
        stripepy.step_2,
        chrom_name=chrom_name,
        chrom_size=chrom_size,
        matrix=lt_matrix,
        min_persistence=min_persistence,
        location="lower",
    )

    task2 = executor.submit(
        stripepy.step_2,
        chrom_name=chrom_name,
        chrom_size=chrom_size,
        matrix=ut_matrix,
        min_persistence=min_persistence,
        location="upper",
    )

    result = _merge_results((task1, task2))

    logger.info("topological data analysis took %s", pretty_format_elapsed_time(t0))
    return result


def run(
    contact_map: pathlib.Path,
    resolution: int,
    output_file: pathlib.Path,
    genomic_belt: int,
    max_width: int,
    glob_pers_min: float,
    constrain_heights: bool,
    loc_pers_min: float,
    loc_trend_min: float,
    force: bool,
    nproc: int,
    min_chrom_size: int,
    verbosity: str,
    main_logger: logging.ProcessSafeLogger,
    roi: Optional[str] = None,
    log_file: Optional[pathlib.Path] = None,
    plot_dir: Optional[pathlib.Path] = None,
    normalization: Optional[str] = None,
) -> int:
    args = locals()
    args.pop("main_logger")
    # How long does stripepy take to analyze the whole Hi-C matrix?
    start_global_time = time.time()

    parent_logger = main_logger
    main_logger = structlog.get_logger().bind(step="main")

    _write_param_summary(args, logger=main_logger)

    if roi is not None:
        # Raise an error immediately if --roi was passed and matplotlib is not available
        _import_matplotlib()

    f = others.open_matrix_file_checked(contact_map, resolution, logger=main_logger)
    chroms = f.chromosomes(include_ALL=False)

    if force:
        _remove_existing_output_files(output_file, plot_dir, chroms)

    with contextlib.ExitStack() as ctx:
        io_manager = ctx.enter_context(
            IOManager(
                matrix_path=contact_map,
                result_path=output_file,
                resolution=resolution,
                normalization=normalization,
                genomic_belt=genomic_belt,
                region_of_interest=roi,
                metadata=_generate_metadata_attribute(
                    constrain_heights=constrain_heights,
                    genomic_belt=genomic_belt,
                    glob_pers_min=glob_pers_min,
                    loc_pers_min=loc_pers_min,
                    loc_trend_min=loc_trend_min,
                    max_width=max_width,
                    min_chrom_size=min_chrom_size,
                ),
                nproc=nproc,
                main_logger=parent_logger,
            )
        )

        disable_bar = not sys.stderr.isatty()
        progress_weights_df = _compute_progress_bar_weights(chroms, include_plotting=roi is not None, nproc=nproc)
        progress_bar = ctx.enter_context(
            initialize_progress_bar(
                total=sum(chroms.values()),
                manual=True,
                disable=disable_bar,
                enrich_print=False,
                file=sys.stderr,
                receipt=False,
                refresh_secs=0.05,
                monitor="{percent:.2%}",
                unit="bp",
                scale="SI",
            )
        )

        if normalization is None:
            normalization = "NONE"

        tpool = _setup_tpool(ctx, nproc, parent_logger)
        tasks = _plan(chroms, min_chrom_size)

        for i, (chrom_name, chrom_size, skip) in enumerate(tasks):
            progress_weights = progress_weights_df.loc[chrom_name, :].to_dict()

            if skip:
                logger.warning("writing an empty entry for chromosome %s...", chrom_name)
                io_manager.write_results(_generate_empty_result(chrom_name, chrom_size, resolution))
                progress_bar(max(progress_weights.values()))
                continue

            logger = main_logger.bind(chrom=chrom_name)
            logger.info("begin processing...")
            start_local_time = time.time()

            LT_Iproc, UT_Iproc, Iproc_RoI = io_manager.fetch_interaction_matrix(chrom_name, chrom_size)
            io_manager.fetch_next_interaction_matrix_async(tasks[i + 1 :])

            with ProcessPoolWrapper(
                nproc=nproc,
                lt_matrix=LT_Iproc,
                ut_matrix=UT_Iproc,
                main_logger=parent_logger,
                init_mpl=roi is not None,
                logger=logger,
            ) as pool:
                if pool.ready:
                    # matrices are stored in the shared global state
                    LT_Iproc = None
                    UT_Iproc = None

                progress_bar(progress_weights["step_1"])
                logger.info("preprocessing took %s", pretty_format_elapsed_time(start_local_time))

                result = _run_step_2(
                    chrom_name=chrom_name,
                    chrom_size=chrom_size,
                    lt_matrix=LT_Iproc,
                    ut_matrix=UT_Iproc,
                    min_persistence=glob_pers_min,
                    tpool=tpool,
                    pool=pool,
                    logger=logger,
                )

                progress_bar(progress_weights["step_2"])

                logger = logger.bind(step=(3,))
                logger.info("shape analysis")
                start_time = time.time()
                task1 = tpool.submit(
                    stripepy.step_3,
                    result,
                    LT_Iproc,
                    resolution,
                    genomic_belt,
                    max_width,
                    loc_pers_min,
                    loc_trend_min,
                    location="lower",
                    map_=pool.map,
                    logger=logger,
                )

                task2 = tpool.submit(
                    stripepy.step_3,
                    result,
                    UT_Iproc,
                    resolution,
                    genomic_belt,
                    max_width,
                    loc_pers_min,
                    loc_trend_min,
                    location="upper",
                    map_=pool.map,
                    logger=logger,
                )

                result = _merge_results((task1, task2))

                progress_bar(progress_weights["step_3"])
                logger.info("shape analysis took %s", pretty_format_elapsed_time(start_time))

                logger = logger.bind(step=(4,))
                logger.info("statistical analysis and post-processing")
                start_time = time.time()

                task1 = tpool.submit(
                    stripepy.step_4,
                    result.get("stripes", "lower"),
                    LT_Iproc,
                    location="lower",
                    map_=pool.map,
                    logger=logger,
                )

                task2 = tpool.submit(
                    stripepy.step_4,
                    result.get("stripes", "upper"),
                    UT_Iproc,
                    location="upper",
                    map_=pool.map,
                    logger=logger,
                )

                result.set("stripes", task1.result()[1], "LT", force=True)
                result.set("stripes", task2.result()[1], "UT", force=True)

                progress_bar(progress_weights["step_4"])
                logger.info("statistical analysis and post-processing took %s", pretty_format_elapsed_time(start_time))

                io_manager.write_results(result)
                progress_bar(progress_weights["output"])
                logger.info("processing took %s", pretty_format_elapsed_time(start_local_time))

                if Iproc_RoI is not None:
                    RoI = others.define_RoI(roi, chrom_size, resolution)
                    logger.info("region of interest to be used for plotting: %s:%d-%d", chrom_name, *RoI["genomic"])
                    result.set_roi(RoI)
                    start_time = time.time()
                    logger = logger.bind(step=(5,))
                    logger.info("generating plots")
                    query = f"{chrom_name}:{result.roi['genomic'][0]}-{result.roi['genomic'][1]}"
                    stripepy.step_5(
                        result,
                        resolution,
                        LT_Iproc,
                        UT_Iproc,
                        f.fetch(query, normalization=normalization).to_numpy("full"),
                        Iproc_RoI,
                        genomic_belt,
                        loc_pers_min,
                        loc_trend_min,
                        plot_dir,
                        pool=pool,
                    )

                    progress_bar(progress_weights["step_5"])
                    logger.info("plotting took %s", pretty_format_elapsed_time(start_time))

        main_logger.bind(step="IO").info('finalizing file "%s"...', output_file)
        io_manager.finalize_results()

    main_logger.info("DONE!")
    main_logger.info("processed %d chromosomes in %s", len(chroms), pretty_format_elapsed_time(start_global_time))

    return 0
