# Copyright (C) 2025 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import collections.abc
import contextlib
import functools
import importlib.util
import multiprocessing as mp
import pathlib
import platform
import sys
from typing import Dict, List, Optional

import hictkpy
import structlog


@functools.cache
def _map_log_level_to_levelno(level: str) -> int:
    levels = {
        "critical": 50,
        "error": 40,
        "warning": 30,
        "info": 20,
        "debug": 10,
        "notset": 0,
    }

    return levels.get(level.lower(), 0)


class _MultiOutputLogger(object):
    def __init__(self, console, file):
        self._console = console
        self._file = file

    def write(self, file_message: Optional[str], console_message: Optional[str]):
        if self._file is not None and file_message is not None:
            print(file_message, file=self._file)
        if self._console is not None and console_message is not None:
            print(console_message, file=self._console, flush=True)

    def flush(self):
        if self._console is not None:
            self._console.flush()
        if self._file is not None:
            self._file.flush()


class _PrintLogger(object):
    def __init__(self, console, file):
        self._logger = _MultiOutputLogger(console, file)

    def msg(self, file_message: Optional[str] = None, console_message: Optional[str] = None):
        self._logger.write(file_message, console_message)

    log = debug = info = warn = warning = msg
    fatal = failure = err = error = critical = exception = msg


class PrintLoggerFactory(object):
    def __init__(self, console, file):
        self._console = console
        self._file = file

    def __call__(self) -> _PrintLogger:
        return _PrintLogger(self._console, self._file)


class _NullLogger(object):
    def msg(self, *args, **kwargs):
        pass

    log = debug = info = warn = warning = msg
    fatal = failure = err = error = critical = exception = msg


class NullLoggerFactory(object):
    def __call__(self) -> _NullLogger:
        return _NullLogger()


class _StructLogPlainStyles(object):
    def __init__(self):
        self.reset = ""
        self.bright = ""
        self.dim = ""

        self.level_critical = ""
        self.level_exception = ""
        self.level_error = ""
        self.level_warn = ""
        self.level_info = ""
        self.level_debug = ""
        self.level_notset = ""

        self.timestamp = ""
        self.chromosome = ""
        self.location = ""
        self.step = ""
        self.logger_name = ""


class _StructLogColorfulStyles(object):
    @staticmethod
    def _try_get_color(key: str):
        try:
            import colorama

            return eval(f"colorama.{key}")
        except ImportError:
            return ""

    def __init__(self):
        _try_get_color = _StructLogColorfulStyles._try_get_color
        self.reset = _try_get_color("Style.RESET_ALL")
        self.bright = _try_get_color("Style.BRIGHT")
        self.dim = _try_get_color("Style.DIM")

        self.level_critical = _try_get_color("Fore.RED")
        self.level_exception = _try_get_color("Fore.RED")
        self.level_error = _try_get_color("Fore.RED")
        self.level_warn = _try_get_color("Fore.YELLOW")
        self.level_info = _try_get_color("Fore.GREEN")
        self.level_debug = _try_get_color("Fore.GREEN")
        self.level_notset = _try_get_color("Back.RED")

        self.timestamp = self.dim
        self.chromosome = _try_get_color("Fore.BLUE")
        self.location = _try_get_color("Fore.BLUE")
        self.step = _try_get_color("Fore.BLUE")
        self.logger_name = _try_get_color("Fore.BLUE")


def _configure_logger_columns(
    colors: bool,
    level_styles: Optional[structlog.dev.Styles] = None,
    event_key: str = "event",
    timestamp_key: str = "timestamp",
    pad_level: bool = True,
    longest_chrom_name: str = "chrXX",
    max_step_nest_levels: int = 3,
) -> List:
    """
    The body of this function is an extension of the structlog.dev.ConsoleRenderer:
    https://github.com/hynek/structlog/blob/a60ce7bbb50451ed786ace3c3893fb3a6a01df0a/src/structlog/dev.py#L433
    """

    if colors and importlib.util.find_spec("colorama") is None:
        return _configure_logger_columns(
            False, level_styles, event_key, timestamp_key, pad_level, longest_chrom_name, max_step_nest_levels
        )

    level_to_color = (
        structlog.dev.ConsoleRenderer().get_default_level_styles(colors) if level_styles is None else level_styles
    )

    if hasattr(structlog.dev, "_EVENT_WIDTH"):
        pad_event = structlog.dev._EVENT_WIDTH  # noqa
    else:
        pad_event = 30

    pad_chrom = len(longest_chrom_name)
    pad_step = len("step ") + max_step_nest_levels + (max_step_nest_levels - 1)

    level_width = 0 if not pad_level else None

    styles: structlog.Styles
    if colors:
        if platform.system() == "Windows":
            # Colorama must be init'd on Windows, but must NOT be
            # init'd on other OSes, because it can break colors.
            import colorama

            colorama.init()

        styles = _StructLogColorfulStyles()
    else:
        styles = _StructLogPlainStyles()

    def step_formatter(data):
        if isinstance(data, str) and (data.startswith("IO") or data == "main"):
            return data
        if isinstance(data, collections.abc.Sequence):
            return f"step {'.'.join(str(x) for x in data)}"
        return f"step {data}"

    return [
        structlog.dev.Column(
            timestamp_key,
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=styles.timestamp,
                reset_style=styles.reset,
                value_repr=str,
            ),
        ),
        structlog.dev.Column(
            "level",
            structlog.dev.LogLevelColumnFormatter(level_to_color, reset_style=styles.reset, width=level_width),
        ),
        structlog.dev.Column(
            "chrom",
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=styles.chromosome,
                reset_style=styles.reset,
                value_repr=str,
                width=pad_chrom,
                prefix="[",
                postfix="]",
            ),
        ),
        structlog.dev.Column(
            "step",
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=styles.step,
                reset_style=styles.reset,
                value_repr=step_formatter,
                width=pad_step,
                prefix="[",
                postfix="]",
            ),
        ),
        structlog.dev.Column(
            "location",
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=styles.location,
                reset_style=styles.reset,
                value_repr=str,
                width=2,
                prefix="[",
                postfix="]",
            ),
        ),
        structlog.dev.Column(
            event_key,
            structlog.dev.KeyValueColumnFormatter(
                key_style=None,
                value_style=styles.bright,
                reset_style=styles.reset,
                value_repr=str,
                width=pad_event,
            ),
        ),
        structlog.dev.Column(
            "",
            structlog.dev.KeyValueColumnFormatter(
                key_style=None, value_style=styles.dim, reset_style=styles.reset, value_repr=str
            ),
        ),
    ]


def _get_longest_chrom_name(path: pathlib.Path) -> str:
    try:
        chroms = hictkpy.MultiResFile(path).chromosomes(include_ALL=False).keys()
    except RuntimeError:
        try:
            chroms = hictkpy.File(path).chromosomes(include_ALL=False).keys()
        except RuntimeError:
            chroms = ("chrXX",)

    return max(chroms, key=len)  # noqa


class ProcessSafeLogger(object):
    def __init__(
        self,
        level: str,
        path: Optional[pathlib.Path],
        force: bool = False,
        matrix_file: Optional[pathlib.Path] = None,
    ):
        self._level = level
        self._path = path
        self._force = force
        self._queue = None
        self._listener = None
        self._log_file = None
        if matrix_file is None:
            self._longest_chrom_name = ""
        else:
            self._longest_chrom_name = _get_longest_chrom_name(matrix_file)

    def __enter__(self):
        self._queue = mp.Manager().Queue(64 * 1024)

        if self._path is not None:
            if self._path.exists() and not self._force:
                raise RuntimeError(
                    f'Refusing to overwrite existing log file "{self._path}". Pass --force to overwrite.'
                )

        self._listener = mp.Process(
            target=ProcessSafeLogger._listener,
            args=(
                self._path,
                self._level,
                "DEBUG",
                self._longest_chrom_name,
                self._queue,
            ),
        )
        self._listener.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._listener is not None:
            self._queue.put(None)
            self._listener.join()
        return False

    @staticmethod
    def _listener(
        path: pathlib.Path,
        log_level_console: str,
        log_level_file: str,
        longest_chrom_name: str,
        queue: mp.Queue,
    ):
        proc = mp.current_process()
        with contextlib.ExitStack() as ctx:
            error = None
            try:
                if path is None:
                    log_file = None
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.unlink(missing_ok=True)
                    log_file = ctx.enter_context(path.open("w", encoding="utf-8"))
            except Exception:  # noqa
                import traceback

                error = traceback.format_exc()
                log_file = None

            ProcessSafeLogger._setup_logger_for_listener(
                log_level_console,
                log_level_file,
                console_handle=sys.stderr,
                file_handle=log_file,
                longest_chrom_name=longest_chrom_name,
            )
            logger = structlog.get_logger()

            if path is not None:
                if error is not None:
                    logger.warn(
                        'failed to initialize log file "%s" for writing:\n%s',
                        path,
                        error,
                    )
                else:
                    assert path.is_file()
                    logger.debug('%s (PID=%d) successfully initialized log file "%s"', proc.name, proc.pid, path)

            log_level_mapper = _map_log_level_to_levelno

            while True:
                event_dict = queue.get()
                if event_dict is None:
                    logger.debug("%s (PID=%d): processed all log messages: returning!", proc.name, proc.pid)
                    return
                event_dict["level"] = log_level_mapper(event_dict.pop("level", "notset"))
                logger.log(**event_dict)

    @property
    def log_queue(self) -> Optional[mp.Queue]:
        return self._queue

    @staticmethod
    def _setup_logger_for_listener(
        log_level_console: str,
        log_level_file: str,
        console_handle,
        file_handle,
        longest_chrom_name,
    ):
        timestamper = structlog.processors.MaybeTimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f")

        def maybe_add_log_level(logger, method_name: str, event_dict):
            if "level" in event_dict:
                return event_dict

            return structlog.processors.add_log_level(logger, method_name, event_dict)

        plain_renderer = structlog.dev.ConsoleRenderer(
            columns=_configure_logger_columns(colors=False, longest_chrom_name=longest_chrom_name),
        )
        if console_handle is None or not console_handle.isatty():
            colored_renderer = None
        else:
            colored_renderer = structlog.dev.ConsoleRenderer(
                columns=_configure_logger_columns(colors=True, longest_chrom_name=longest_chrom_name),
            )

        log_level_mapper = _map_log_level_to_levelno
        console_levelno = log_level_mapper(log_level_console)
        file_levelno = log_level_mapper(log_level_file)

        def renderer(logger, name: str, event_dict) -> Dict[Optional[str], Optional[str]]:
            file_message = None
            console_message = None

            log_lvl = log_level_mapper(event_dict["level"])

            if file_levelno <= log_lvl:
                file_message = plain_renderer(logger, name, event_dict.copy())

            if console_levelno <= log_level_mapper(event_dict["level"]):
                if colored_renderer is not None:
                    console_message = colored_renderer(logger, name, event_dict)
                elif file_message is None:
                    console_message = plain_renderer(logger, name, event_dict)
                else:
                    console_message = file_message

            return {"file_message": file_message, "console_message": console_message}

        processors = [
            timestamper,
            maybe_add_log_level,
            structlog.processors.StackInfoRenderer(),
            renderer,
        ]

        structlog.configure(
            cache_logger_on_first_use=True,
            wrapper_class=structlog.make_filtering_bound_logger(0),
            processors=processors,
            logger_factory=PrintLoggerFactory(console=console_handle, file=file_handle),
        )

    def setup_logger(self):
        ProcessSafeLogger._setup_logger(self._queue)

    @staticmethod
    def _queue_logger_helper(_, method_name, event_dict, queue: mp.Queue) -> str:
        queue.put_nowait(event_dict)
        return ""

    @staticmethod
    def _queue_logger(queue: mp.Queue):
        return functools.partial(ProcessSafeLogger._queue_logger_helper, queue=queue)

    @staticmethod
    def _setup_logger(queue: mp.Queue):
        timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S.%f")
        processors = [
            timestamper,
            structlog.processors.add_log_level,
            ProcessSafeLogger._queue_logger(queue),
        ]

        structlog.configure(
            cache_logger_on_first_use=True,
            processors=processors,
            wrapper_class=None,
            logger_factory=NullLoggerFactory(),
        )

        proc = mp.current_process()
        structlog.get_logger().debug("successfully initialized logger in %s with PID=%d", proc.name, proc.pid)
