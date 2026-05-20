import numpy as np
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score
from src.ml.split import classic_train_test_split
from src.ml.tuning import compute_and_save_best_parameters
from src.ml.models import train_and_predict
from src.processing.general_processing_functions import save_dict_json
from src.core.config import BETA, REL_THR
from src.core.logger import logger


def evaluate_predictions(y_test, y_pred):
    """
    Evaluate the predictions using accuracy, precision, recall, and F-beta score.
    """

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    fbeta = fbeta_score(y_test, y_pred, beta=BETA)

    return {"accuracy": acc, "precision": prec, "recall": rec, "fbeta": fbeta}


def find_best_threshold(y_true, y_proba):
    """
    Find the best threshold for converting probabilities to binary predictions based on F-beta score.
    """

    thresholds = np.arange(0, 1, 0.01)
    scores = [fbeta_score(y_true, (y_proba >= t).astype(int), beta=BETA) for t in thresholds]
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]

    y_pred_optimal = (y_proba >= best_threshold).astype(int)
    metrics = evaluate_predictions(y_true, y_pred_optimal)

    return best_threshold, metrics


def filter_players_by_reliability(df_features):
    """
    Filter players based on their reliability score.
    """

    df_filtered = df_features[df_features['reliability_score'] >= REL_THR].reset_index(drop=True)

    return df_filtered


def models_evaluation(df_features, save_results=False):
    """
    Evaluate the performance of different models on the given features and save the results if specified.
    """
    
    # Filter players based on reliability
    df = filter_players_by_reliability(df_features)

    # Split the data by classic train-test split
    X_train, X_test, y_train, y_test = classic_train_test_split(df)

    # Compute and save the best parameters on the training set for each model
    best_params = compute_and_save_best_parameters(X_train, y_train)

    # Train and predict with each model
    y_lr, y_proba_lr = train_and_predict(X_train, y_train, X_test, "logistic_regression")
    y_rf, y_proba_rf = train_and_predict(X_train, y_train, X_test, "random_forest")
    y_xgb, y_proba_xgb = train_and_predict(X_train, y_train, X_test, "xgboost")

    # Evaluate predictions for each model
    metrics_lr = evaluate_predictions(y_test, y_lr)
    metrics_rf = evaluate_predictions(y_test, y_rf)
    metrics_xgb = evaluate_predictions(y_test, y_xgb)

    logger.info(f"Logistic Regression - Accuracy: {metrics_lr['accuracy']:.2f}, Precision: {metrics_lr['precision']:.2f}, Recall: {metrics_lr['recall']:.2f}, F-BETA: {metrics_lr['fbeta']:.2f}")
    logger.info(f"Random Forest - Accuracy: {metrics_rf['accuracy']:.2f}, Precision: {metrics_rf['precision']:.2f}, Recall: {metrics_rf['recall']:.2f}, F-BETA: {metrics_rf['fbeta']:.2f}")
    logger.info(f"XGBoost - Accuracy: {metrics_xgb['accuracy']:.2f}, Precision: {metrics_xgb['precision']:.2f}, Recall: {metrics_xgb['recall']:.2f}, F-BETA: {metrics_xgb['fbeta']:.2f}")

    # Find the best threshold for each model and evaluate the metrics at that threshold
    thr_lr, scores_lr = find_best_threshold(y_test, y_proba_lr)
    thr_rf, scores_rf = find_best_threshold(y_test, y_proba_rf)
    thr_xgb, scores_xgb = find_best_threshold(y_test, y_proba_xgb)

    logger.info(f"Logistic Regression - Best Threshold: {thr_lr:.2f}, Accuracy: {scores_lr['accuracy']:.2f}, Precision: {scores_lr['precision']:.2f}, Recall: {scores_lr['recall']:.2f}, F-BETA: {scores_lr['fbeta']:.2f}")
    logger.info(f"Random Forest - Best Threshold: {thr_rf:.2f}, Accuracy: {scores_rf['accuracy']:.2f}, Precision: {scores_rf['precision']:.2f}, Recall: {scores_rf['recall']:.2f}, F-BETA: {scores_rf['fbeta']:.2f}")
    logger.info(f"XGBoost - Best Threshold: {thr_xgb:.2f}, Accuracy: {scores_xgb['accuracy']:.2f}, Precision: {scores_xgb['precision']:.2f}, Recall: {scores_xgb['recall']:.2f}, F-BETA: {scores_xgb['fbeta']:.2f}")

    if save_results:
        results = {
            "logistic_regression": {"metrics": metrics_lr, "best_threshold": thr_lr, "threshold_metrics": scores_lr},
            "random_forest": {"metrics": metrics_rf, "best_threshold": thr_rf, "threshold_metrics": scores_rf},
            "xgboost": {"metrics": metrics_xgb, "best_threshold": thr_xgb, "threshold_metrics": scores_xgb}
        }
        save_dict_json(results, "model_evaluation_results")
