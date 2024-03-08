import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
import os
from .preprocess import clean_data
from .preprocess import fill_missing_values
from .preprocess import feature_selection
from .preprocess import load_features
from .preprocess import preprocess_features
from .preprocess import split_data


def train_model(X_train: np.array, y_train: pd.Series,
                model_dir: str) -> XGBRegressor:
    param = {
        'max_depth': 4,
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'eval_metric': 'rmse'
    }
    model = XGBRegressor(**param)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(model_dir, 'model.joblib'))
    print(f'Model saved to {os.path.join(model_dir, "model.joblib")}')


# function for calucalting Root Mean Squared Logarithmic Error
def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray,
                  precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


# function for calculating the other metrices
def evaluate_performance(y_test, y_pred):
    rmsle_score = compute_rmsle(np.log(y_test), np.log(y_pred))
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = r2_score(y_test, y_pred)
    evaluation_results = {
        "rmsle_score": rmsle_score,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R^2": r2
                }
    return evaluation_results


def model_training(X_train, y_train, model_dir):
    X_train = clean_data(X_train)
    X_train = fill_missing_values(X_train)
    top_10_numerical_features, top_5_categorical_features =\
        feature_selection(X_train, y_train)
    X_train_processed = preprocess_features(
        X_train,
        top_10_numerical_features,
        top_5_categorical_features,
        model_dir,
        'train')
    train_model(X_train_processed, y_train, model_dir)


def model_evaluation(X_test, y_test, model_dir):
    X_test = clean_data(X_test)
    X_test = fill_missing_values(X_test)
    # Specify the file path from where you want to load the JSON
    feature_file_path = ('/home/sachin/DSP/dsp-anandhu-krishna/'
                         'models/features_dictionary.json')
    top_10_numerical_features, top_5_categorical_features = \
        load_features(feature_file_path)
    X_test_processed = preprocess_features(
        X_test,
        top_10_numerical_features,
        top_5_categorical_features,
        model_dir,
        'test')
    model_path = os.path.join(model_dir, 'model.joblib')
    # model loading
    model = joblib.load(model_path)
    # model predicitng
    y_pred = model.predict(X_test_processed)

    evaluation_results = evaluate_performance(y_test, y_pred)
    return evaluation_results


def build_model(data: pd.DataFrame) -> dict[str, str]:
    model_dir = '/home/sachin/DSP/dsp-anandhu-krishna/models'
    X_train, X_test, y_train, y_test = split_data(data, "SalePrice")
    model_training(X_train, y_train, model_dir)
    performances = model_evaluation(X_test, y_test, model_dir)
    return performances
