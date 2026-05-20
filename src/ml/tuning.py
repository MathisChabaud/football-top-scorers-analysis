import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score, make_scorer
from src.processing.general_processing_functions import save_dict_json
from src.ml.models import get_model
from src.core.config import BETA


def run_search(pipeline, params, X, y):
    """
    Run GridSearchCV for a given pipeline and parameter grid.
    """
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'fbeta': make_scorer(fbeta_score, beta=BETA)
    }

    # We search best parameters based on F-BETA score
    search = GridSearchCV(
        pipeline,
        param_grid=params,
        scoring=scoring,
        refit='fbeta',
        cv=cv,
        verbose=1,
        n_jobs=-1,
    )
        
    search.fit(X, y)
        
    return search


def find_best_parameters(X_train, y_train):
    """
    Find the best parameters for each model using GridSearchCV and return the results in a DataFrame.
    """

    pipelines = {
        'logistic_regression': Pipeline([('scaler', StandardScaler()), 
                                         ('model', get_model('logistic_regression', {}))]),
        'random_forest': Pipeline([('model', get_model('random_forest', {}))]),
        'xgboost': Pipeline([('model', get_model('xgboost', {}))])
    }

    param_grids = {
        'logistic_regression': {
            'model__C': [0.01, 0.1, 1, 10, 100],
            'model__penalty': ['l2'],
            'model__solver': ['lbfgs']
        },
        'random_forest': {
            'model__n_estimators': [100, 300, 500],
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt']
        },
        'xgboost': {
            'model__n_estimators': [100, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__subsample': [0.8, 0.9, 1.0],
            'model__colsample_bytree': [0.8, 0.9, 1.0]
        }
    }

    searches = {}
    for name in pipelines:
        searches[name] = run_search(pipelines[name], param_grids[name], X_train, y_train)

    results = pd.DataFrame({
        'Model': ['LogReg', 'RandomForest', 'XGBoost'],
        'Best Params': [searches['logistic_regression'].best_params_,
                        searches['random_forest'].best_params_,
                        searches['xgboost'].best_params_],
        'F-BETA': [searches['logistic_regression'].best_score_,
                   searches['random_forest'].best_score_,
                   searches['xgboost'].best_score_]
    })

    best_params = {}
    for name in searches:
        best_params[name] = {k.replace('model__', ''): v 
                            for k, v in searches[name].best_params_.items()}

    return results, best_params


def compute_and_save_best_parameters(X, y):
    """
    Compute the best parameters for each model and save them in a JSON file.
    """

    results, best_params = find_best_parameters(X, y)

    # Sauvegarder dans un fichier JSON
    save_dict_json(best_params, "best_ml_params")

    return best_params
