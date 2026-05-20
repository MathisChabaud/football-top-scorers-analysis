from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from src.processing.general_processing_functions import load_dict_json


def get_model(model_name, params):
    """
    Get the machine learning model based on the model name and parameters.
    
    model_name: str, name of the model to use. Options: 'logistic_regression', 'random_forest', 'xgboost'.
    params: dict, parameters for the model.
    """

    if model_name == "logistic_regression":
        model = LogisticRegression(**params, max_iter=1000)
    
    elif model_name == "random_forest":
        model = RandomForestClassifier(**params, random_state=42)

    elif model_name == "xgboost":
        model = XGBClassifier(**params, random_state=42, n_jobs=-1)

    else:
        raise ValueError(f"Model name {model_name} not recognized. Choose from 'logistic_regression', 'random_forest', 'xgboost'.")

    return model


def scale_features(X_train, X_test):

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def predict(model_trained, X_test):
    """
    Predict the target variable using the trained model.
    """

    y_pred = model_trained.predict(X_test)
    y_pred_proba = model_trained.predict_proba(X_test)[:, 1]

    return y_pred, y_pred_proba


def train_and_predict(X_train, y_train, new_X, model_name, params_dictname="best_ml_params"):
    """
    Train the specified model and predict the target variable for new data.
    Suppose that the best parameters for the models are stored in a JSON file, and we load them based on the model name.
    """

    all_params = load_dict_json(params_dictname)
    params = all_params.get(model_name)

    if model_name == "logistic_regression":
        X_train, new_X = scale_features(X_train, new_X)
    
    model = get_model(model_name, params)

    model.fit(X_train, y_train)
    y_pred, y_proba = predict(model, new_X)

    return y_pred, y_proba
