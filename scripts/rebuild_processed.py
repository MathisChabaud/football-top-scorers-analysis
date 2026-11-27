import argparse

from src.pipeline import check_rebuild_inputs, planned_rebuild_writes, rebuild
from src.core.logger import logger


def main():

    parser = argparse.ArgumentParser(
        description="Rebuild data/processed/ from the committed raw data, offline."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be written, then exit without writing.",
    )
    args = parser.parse_args()

    check_rebuild_inputs()

    if args.dry_run:
        logger.info("Dry run: nothing will be written.")
        for output, action in planned_rebuild_writes():
            logger.info("  would %s %s", action, output)
        return

    rebuild()

    logger.info("data/processed/ rebuilt. No network access was used.")


if __name__ == "__main__":
    main()
