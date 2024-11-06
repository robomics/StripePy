from .cli import call, download, setup


def main():
    subcommand, args = setup.parse_args()

    if subcommand == "call":
        return call.run(**args)
    if subcommand == "download":
        return download.run(**args)

    raise NotImplementedError


if __name__ == "__main__":
    main()
