import pandas as pd
from src.ml.evaluation import models_evaluation
from src.core.config import PROCESSED_DATA_DIR
from src.core.logger import logger


def main():

    logger.info("Starting model evaluation...")
    df_features = pd.read_csv(f"{PROCESSED_DATA_DIR}/clean_features_dataset.csv")
    models_evaluation(df_features, save_results=True)


if __name__ == "__main__":
    main()