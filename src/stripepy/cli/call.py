# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import concurrent.futures
import contextlib
import functools
import json
import multiprocessing as mp
import pathlib
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
from stripepy.utils import stripe
from stripepy.utils.common import (
    _import_matplotlib,
    pretty_format_elapsed_time,
    zero_columns,
    zero_rows,
)
from stripepy.utils.multiprocess_sparse_matrix import (
    SharedTriangularSparseMatrix,
    SparseMatrix,
    get_shared_state,
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
    lower_triangular_matrix: Optional[SharedTriangularSparseMatrix],
    upper_triangular_matrix: Optional[SharedTriangularSparseMatrix],
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
        init_mpl: bool = False,
        lazy_pool_initialization: bool = False,
        logger=None,
    ):
        self._nproc = nproc
        self._pool = None
        self._lt_matrix = None
        self._ut_matrix = None
        self._init_mpl = init_mpl
        self._log_queue = None

        if nproc > 1:
            self._lock = mp.Lock()
            self._log_queue = main_logger.log_queue
            if not lazy_pool_initialization:
                self._initialize_pool(logger)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pool is None:
            return False

        if exc_type is not None:
            structlog.get_logger().debug("shutting down process pool due to the following exception: %s", exc_val)

        self._pool.shutdown(wait=True, cancel_futures=exc_type is not None)
        if self._lt_matrix is not None:
            unset_shared_state()
            self._lt_matrix = None
            self._ut_matrix = None

        self._pool = None
        return False

    def _can_rebind_shared_matrix(self, ut_matrix: SparseMatrix) -> bool:
        return self._ut_matrix is not None and self._ut_matrix.can_assign(ut_matrix)

    def rebind_shared_matrices(
        self,
        chrom: str,
        ut_matrix: SparseMatrix,
        logger=None,
        max_nnz: Optional[int] = None,
    ):
        if self._nproc < 2:
            return

        if self._can_rebind_shared_matrix(ut_matrix):
            self._ut_matrix.assign(chrom, ut_matrix, logger)
            self._lt_matrix = self._ut_matrix.T

            set_shared_state(self._lt_matrix, self._ut_matrix)
            return

        if max_nnz is not None:
            max_nnz = max(max_nnz, ut_matrix.nnz)

        unset_shared_state()
        self._ut_matrix = (
            SharedTriangularSparseMatrix(chrom, ut_matrix, logger, max_nnz) if ut_matrix is not None else None
        )
        self._lt_matrix = self._ut_matrix.T if self._ut_matrix is not None else None
        set_shared_state(self._lt_matrix, self._ut_matrix)
        self._initialize_pool(logger)

    def _initialize_pool(
        self,
        logger=None,
    ):
        if self._nproc < 2:
            return

        t0 = time.time()
        if logger is not None:
            prefix = "re-" if self._pool is not None else ""
            logger.debug("%sinitializing a process pool of size %d", prefix, self._nproc)

        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)

        self._pool = concurrent.futures.ProcessPoolExecutor(  # noqa
            max_workers=self._nproc,
            initializer=_init_shared_state,
            initargs=(self._lt_matrix, self._ut_matrix, self._log_queue, self._init_mpl),
        )

        if logger is not None:
            logger.debug("pool initialization took %s", pretty_format_elapsed_time(t0))

    @property
    def map(self, chunksize: int = 1):
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
        self._tasks = {}

        logger = structlog.get_logger().bind(step="IO")

        self._pool = ProcessPoolWrapper(
            min(nproc, 2),
            main_logger=main_logger,
            init_mpl=False,
            lazy_pool_initialization=False,
            logger=logger,
        )

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
        self._pool.__exit__(exc_type, exc_val, exc_tb)

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
        ut_matrix, roi_matrix_raw, roi_matrix_proc = stripepy.step_1(
            matrix,
            genomic_belt,
            resolution,
            roi=roi,
            logger=logger,
        )
        logger.info("preprocessing took %s", pretty_format_elapsed_time(t0))

        return ut_matrix, roi_matrix_raw, roi_matrix_proc

    def fetch_interaction_matrix(
        self, chrom_name: str, chrom_size: int
    ) -> Tuple[SparseMatrix, Optional[SparseMatrix], Optional[SparseMatrix]]:
        data = self.get_interaction_matrix(chrom_name)
        if data is not None:
            structlog.get_logger().bind(chrom=chrom_name, step="IO").info("returning pre-fetched interactions")
            return data

        roi = others.define_RoI(self._roi, chrom_size, self._resolution)
        return IOManager._fetch(  # noqa
            self._path,
            self._resolution,
            self._normalization,
            self._genomic_belt,
            chrom_name,
            roi,
        )

    def fetch_interaction_matrix_async(self, chrom_name: str, chrom_size: int):
        if not self._pool.ready:
            return

        assert chrom_name not in self._tasks

        roi = others.define_RoI(self._roi, chrom_size, self._resolution)
        self._tasks[chrom_name] = self._pool.submit(
            IOManager._fetch,
            self._path,
            self._resolution,
            self._normalization,
            self._genomic_belt,
            chrom_name,
            roi,
        )

    def fetch_next_interaction_matrix_async(self, tasks: List[Tuple[str, int, bool]]):
        if not self._pool.ready:
            return

        if len(tasks) == 0 or tasks[0][0] in self._tasks:
            return

        for chrom, size, skip in tasks:
            if not skip:
                self.fetch_interaction_matrix_async(chrom, size)
                return

    def get_interaction_matrix(
        self, chrom: str
    ) -> Optional[Tuple[SparseMatrix, Optional[SparseMatrix], Optional[SparseMatrix]]]:
        res = self._tasks.pop(chrom, None)
        if res is not None:
            res = res.result()  # noqa
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
        self._h5_pending_io_task = self._pool.submit(IOManager._write_results, self._h5_path, result)

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
    for i, res in enumerate(futures, 1):
        if isinstance(res, concurrent.futures.Future):
            res = res.result()

        location, result = res
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


def _run_step_2_helper(args) -> Tuple[str, IO.Result]:
    return stripepy.step_2(*args)


def _run_step_3_helper(args) -> Tuple[str, IO.Result]:
    return stripepy.step_3(*args)


def _run_step_4_helper(args) -> Tuple[str, List[stripe.Stripe]]:
    return stripepy.step_4(*args)


def _fetch_matrix_metadata(lt_matrix: Optional[SparseMatrix], ut_matrix: Optional[SparseMatrix]) -> Tuple[Dict, Dict]:
    if lt_matrix is None:
        lt_matrix_metadata = get_shared_state("lower").metadata
    else:
        lt_matrix_metadata = None
    if ut_matrix is None:
        ut_matrix_metadata = get_shared_state("upper").metadata
    else:
        ut_matrix_metadata = None

    return lt_matrix_metadata, ut_matrix_metadata


def _run_step_2(
    chrom_name: str,
    chrom_size: int,
    lt_matrix: Optional[SparseMatrix],
    ut_matrix: Optional[SparseMatrix],
    min_persistence: float,
    pool: ProcessPoolWrapper,
    logger,
) -> IO.Result:
    logger = logger.bind(step=(2,))
    logger.info("topological data analysis")
    t0 = time.time()

    lt_matrix_metadata, ut_matrix_metadata = _fetch_matrix_metadata(lt_matrix, ut_matrix)

    params = (
        (chrom_name, chrom_size, lt_matrix, lt_matrix_metadata, min_persistence, "lower"),
        (chrom_name, chrom_size, ut_matrix, ut_matrix_metadata, min_persistence, "upper"),
    )

    tasks = pool.map(_run_step_2_helper, params)
    result = _merge_results(tasks)

    logger.info("topological data analysis took %s", pretty_format_elapsed_time(t0))
    return result


def _run_step_3(
    result: IO.Result,
    lt_matrix: Optional[SparseMatrix],
    ut_matrix: Optional[SparseMatrix],
    resolution: int,
    genomic_belt: int,
    max_width: int,
    loc_pers_min: float,
    loc_trend_min: float,
    tpool: Union[ProcessPoolWrapper, concurrent.futures.ThreadPoolExecutor],
    pool: ProcessPoolWrapper,
    logger,
) -> IO.Result:
    logger = logger.bind(step=(3,))
    logger.info("shape analysis")
    t0 = time.time()

    if pool.map is map:
        executor = pool.map
    else:
        executor = functools.partial(pool.map, chunksize=50)

    lt_matrix_metadata, ut_matrix_metadata = _fetch_matrix_metadata(lt_matrix, ut_matrix)

    params = (
        (
            result,
            lt_matrix,
            lt_matrix_metadata,
            resolution,
            genomic_belt,
            max_width,
            loc_pers_min,
            loc_trend_min,
            "lower",
            executor,
            logger,
        ),
        (
            result,
            ut_matrix,
            ut_matrix_metadata,
            resolution,
            genomic_belt,
            max_width,
            loc_pers_min,
            loc_trend_min,
            "upper",
            executor,
            logger,
        ),
    )

    tasks = tpool.map(_run_step_3_helper, params)
    result = _merge_results(tasks)

    logger.info("shape analysis took %s", pretty_format_elapsed_time(t0))

    return result


def _run_step_4(
    result: IO.Result,
    lt_matrix: Optional[SparseMatrix],
    ut_matrix: Optional[SparseMatrix],
    tpool: Union[ProcessPoolWrapper, concurrent.futures.ThreadPoolExecutor],
    pool: ProcessPoolWrapper,
    logger,
) -> IO.Result:
    t0 = time.time()

    logger = logger.bind(step=(4,))
    logger.info("statistical analysis and post-processing")

    if pool.map is map:
        executor = pool.map
    else:
        executor = functools.partial(pool.map, chunksize=50)

    lt_matrix_metadata, ut_matrix_metadata = _fetch_matrix_metadata(lt_matrix, ut_matrix)

    params = (
        (result.get("stripes", "lower"), lt_matrix, lt_matrix_metadata, "lower", executor, logger),
        (result.get("stripes", "upper"), ut_matrix, ut_matrix_metadata, "upper", executor, logger),
    )

    (_, lt_stripes), (_, ut_stripes) = list(tpool.map(_run_step_4_helper, params))
    result.set("stripes", lt_stripes, "LT", force=True)
    result.set("stripes", ut_stripes, "UT", force=True)

    logger.info("statistical analysis and post-processing took %s", pretty_format_elapsed_time(t0))

    return result


def _estimate_max_nnz(chrom: str, matrix: SparseMatrix, chroms: Dict[str, int], multiplier: float = 2.0) -> int:
    assert multiplier >= 1
    longest_chrom = max(chroms, key=chroms.get)  # noqa
    if longest_chrom == chrom:
        return int(multiplier * matrix.nnz)

    max_size = chroms[longest_chrom]
    current_size = chroms[chrom]

    ratio = max_size / current_size

    return int(multiplier * ratio * matrix.nnz)


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

        pool = ctx.enter_context(
            ProcessPoolWrapper(
                nproc=nproc,
                main_logger=parent_logger,
                init_mpl=roi is not None,
                lazy_pool_initialization=True,
                logger=None,
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

            ut_matrix, matrix_roi_raw, matrix_roi_proc = io_manager.fetch_interaction_matrix(chrom_name, chrom_size)

            io_manager.fetch_next_interaction_matrix_async(tasks[i + 1 :])
            if i == 0:
                max_nnz = _estimate_max_nnz(chrom_name, ut_matrix, chroms)
                pool.rebind_shared_matrices(chrom_name, ut_matrix, logger, max_nnz)
            else:
                pool.rebind_shared_matrices(chrom_name, ut_matrix, logger, max_nnz)

            if pool.ready:
                # matrices are stored in the shared global state
                lt_matrix = None
                ut_matrix = None
            else:
                lt_matrix = ut_matrix.T

            progress_bar(progress_weights["step_1"])

            result = _run_step_2(
                chrom_name=chrom_name,
                chrom_size=chrom_size,
                lt_matrix=lt_matrix,
                ut_matrix=ut_matrix,
                min_persistence=glob_pers_min,
                pool=pool,
                logger=logger,
            )
            progress_bar(progress_weights["step_2"])

            result = _run_step_3(
                result=result,
                lt_matrix=lt_matrix,
                ut_matrix=ut_matrix,
                resolution=resolution,
                genomic_belt=genomic_belt,
                max_width=max_width,
                loc_pers_min=loc_pers_min,
                loc_trend_min=loc_trend_min,
                tpool=tpool,
                pool=pool,
                logger=logger,
            )
            progress_bar(progress_weights["step_3"])

            result = _run_step_4(
                result=result,
                lt_matrix=lt_matrix,
                ut_matrix=ut_matrix,
                tpool=tpool,
                pool=pool,
                logger=logger,
            )
            progress_bar(progress_weights["step_4"])

            logger.info("processing took %s", pretty_format_elapsed_time(start_local_time))

            io_manager.write_results(result)
            progress_bar(progress_weights["output"])

            if matrix_roi_raw is not None:
                assert matrix_roi_proc is not None
                chrom_roi = others.define_RoI(roi, chrom_size, resolution)
                logger.info("region of interest to be used for plotting: %s:%d-%d", chrom_name, *chrom_roi["genomic"])
                result.set_roi(chrom_roi)
                start_time = time.time()
                logger = logger.bind(step=(5,))
                logger.info("generating plots")
                stripepy.step_5(
                    result,
                    resolution,
                    matrix_roi_raw,
                    matrix_roi_proc,
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
