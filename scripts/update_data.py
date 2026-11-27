from src.pipeline import SCRAPE_STEPS, REBUILD_STEPS, scrape, rebuild
from src.core.logger import logger


def main():

    logger.info("=== Scraping Understat (%d steps) ===", len(SCRAPE_STEPS))
    scrape()

    logger.info("=== Rebuilding data/processed/ (%d steps) ===", len(REBUILD_STEPS))
    rebuild()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
