from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from aeosbench.cli.generate_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())

