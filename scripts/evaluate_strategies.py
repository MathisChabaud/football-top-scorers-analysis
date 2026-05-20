from src.betting.test_strategy import create_results_dataset, create_results_dataset_for_ml_strat
from src.core.logger import logger


def main():
    logger.info("Creating results dataset...")
    create_results_dataset()
    create_results_dataset_for_ml_strat()


if __name__ == "__main__":
    main()