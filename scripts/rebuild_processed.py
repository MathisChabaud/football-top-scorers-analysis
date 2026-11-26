import argparse
import os

from scripts.update_data import REBUILD_STEPS, rebuild
from src.core.config import RAW_DATA_DIR, DICTS_DIR
from src.core.logger import logger


# Files this script reads. All of them are committed to git.
REQUIRED_INPUTS = [
    f"{RAW_DATA_DIR}/all_players_data.csv",
    f"{RAW_DATA_DIR}/goals_top50_last3seasons.csv",
    f"{RAW_DATA_DIR}/matches_top50_last3seasons.csv",
    f"{DICTS_DIR}/match_league_map.json",
    f"{DICTS_DIR}/team_id_mapping.json",
    f"{DICTS_DIR}/all_team_metrics_before_match.json",
]


def check_inputs():
    """
    Fail early with an explicit message if an input file is missing.

    Without this, a missing file surfaces as a bare FileNotFoundError from deep
    inside pandas, three steps later.
    """

    missing = [path for path in REQUIRED_INPUTS if not os.path.isfile(path)]

    if missing:
        raise FileNotFoundError(
            "Missing input file(s):\n  "
            + "\n  ".join(missing)
            + "\n\nThese paths are relative, so this script must be run from the "
            "repository root:\n    python -m scripts.rebuild_processed\n"
            "If the files are genuinely absent, they are tracked by git: check "
            "`git status` and `git checkout -- data/`."
        )


def planned_writes():
    """
    Return (output path, "create" or "overwrite") for every step of the rebuild.
    """

    plan = []
    for _, _, outputs in REBUILD_STEPS:
        for output in outputs:
            plan.append((output, "overwrite" if os.path.isfile(output) else "create"))

    return plan


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

    check_inputs()

    if args.dry_run:
        logger.info("Dry run: nothing will be written.")
        for output, action in planned_writes():
            logger.info("  would %s %s", action, output)
        return

    rebuild()

    logger.info("data/processed/ rebuilt. No network access was used.")


if __name__ == "__main__":
    main()
