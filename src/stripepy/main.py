from .cli import call, setup


def main():
    subcommand, args = setup.parse_args()

    if subcommand == "call":
        return call.run(**args)

    raise NotImplementedError


if __name__ == "__main__":
    main()
