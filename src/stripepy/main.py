import logging
import pathlib

from .cli import call, download, setup


def _setup_logger(level: str, file: pathlib.Path | None = None):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    try:
        if file is not None:
            file.parent.mkdir(parents=True, exist_ok=True)

        # TODO is it ok to overwrite existing logs?
        logging.basicConfig(filename=file, level=level, format=fmt)
        logging.getLogger().setLevel(level)
    except Exception as e:  # noqa

        logging.basicConfig(level=level, format=fmt)
        logging.getLogger().setLevel(level)

        if file is not None:
            logging.warning('failed to initialize log file "%s" for writing: %s', file, e)


def main():
    subcommand, args = setup.parse_args()
    verbosity = args["verbosity"].upper()

    if subcommand == "call":
        _setup_logger(verbosity, args["output_folder"] / "log.txt")
        return call.run(**args)
    if subcommand == "download":
        _setup_logger(verbosity)
        return download.run(**args)

    raise NotImplementedError


if __name__ == "__main__":
    main()
