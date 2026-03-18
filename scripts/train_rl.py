from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from aeosbench.cli.train_rl import main


if __name__ == "__main__":
    raise SystemExit(main())

