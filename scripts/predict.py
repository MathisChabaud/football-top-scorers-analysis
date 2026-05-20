import pandas as pd
from src.ml.evaluation import filter_players_by_reliability
from src.ml.models import train_and_predict
from src.processing.ml_features_processing import load_and_prepare_new_data
from src.core.logger import logger
from src.core.config import PROCESSED_DATA_DIR


PLAYER_ID = 3423 # replace with the actual player ID you want to predict for
MODEL_NAME = "logistic_regression"  # or "random_forest", "xgboost"


def main():

    df_features = pd.read_csv(f"{PROCESSED_DATA_DIR}/clean_features_dataset.csv")
    df_train = filter_players_by_reliability(df_features)
    X_train, y_train = df_train.drop(columns=["scored"]), df_train["scored"]
    new_X = load_and_prepare_new_data(player_id=PLAYER_ID)

    model_name = MODEL_NAME
    y_pred, y_proba = train_and_predict(X_train, y_train, new_X, model_name)

    logger.info("Prediction: %s", y_pred)
    logger.info("Probability: %s", y_proba)


if __name__ == "__main__":
    main()
