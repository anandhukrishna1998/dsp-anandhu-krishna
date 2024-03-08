import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# funtion for spliting the data
def split_data(data: pd.DataFrame, target_column, test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# funtion for dropping unwanted columns in the data
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature',
                       'FireplaceQu', 'Id',
                       'GarageYrBlt', '1stFlrSF', 'TotRmsAbvGrd', 'GarageArea']
    data.drop(columns=columns_to_drop, axis=1, inplace=True)
    return data


# funtion for cleaning  the data
def fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numerical_cols:
        data[col] = data[col].fillna(data[col].mean())
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    return data


def feature_selection(X_train, y_train):
    """
    Performs feature selection based on correlation and chi-square test.
    """
    features_dictionary = {}
    train_data = pd.concat([X_train, y_train], axis=1)
    numerical_correlations = train_data.select_dtypes(
        include=[np.number]).corr()['SalePrice'].abs().sort_values(
            ascending=False)
    top_10_numerical_features = numerical_correlations[1:11].index.tolist()
    categorical_features = train_data.select_dtypes(
        exclude=[np.number]).columns.tolist()
    chi2_results = {feature: chi2_contingency(pd.crosstab(
        train_data[feature], train_data['SalePrice']))[0]
        for feature in categorical_features}
    top_5_categorical_features = sorted(
        chi2_results, key=chi2_results.get, reverse=True)[:5]
    print('top_5_categorical_features is ', top_5_categorical_features)
    feature_file_path = ('/home/sachin/DSP/dsp-anandhu-krishna/'
                         'models/features_dictionary.json')
    features_dictionary['top_10_numerical_features'] = \
        top_10_numerical_features
    features_dictionary['top_5_categorical_features'] = \
        top_5_categorical_features
    # Dumping the features dictionary into a JSON file
    with open(feature_file_path, 'w') as file:
        json.dump(features_dictionary, file)
    print(f" features Dictionary saved to {feature_file_path}")
    return top_10_numerical_features, top_5_categorical_features


# save the encoder and decoder objects
def save_preprocessing_objects(encoder, scaler, model_dir: str) -> None:
    joblib.dump(encoder, os.path.join(model_dir, 'encoder.joblib'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
    print(f'Encoder and Scaler saved to {model_dir}')


def load_features(feature_file_path):
    """
    Load the top 10 numerical features and top 5 categorical features from a
    JSON file.

    Parameters:
    feature_file_path (str): The path to the JSON file.

    Returns:
    tuple: A tuple containing two lists: top_10_numerical_features and
    top_5_categorical_features.
    """
    # Loading the JSON data back into a Python dictionary
    with open(feature_file_path, 'r') as file:
        loaded_features_dictionary = json.load(file)
    # Extracting the features from the dictionary
    top_10_numerical_features = \
        loaded_features_dictionary['top_10_numerical_features']

    top_5_categorical_features = \
        loaded_features_dictionary['top_5_categorical_features']
    return top_10_numerical_features, top_5_categorical_features


# def preprocess_features_train(
#     unprocessed_data: pd.DataFrame,
#     numeric_features: list,
#     categorical_features: list,
#     model_dir: str
# ) -> np.ndarray:

#     """
#     Preprocesses the given DataFrame by encoding categorical features and
#     scaling numerical features.
#     Args:
#     - unprocessed_data: DataFrame containing the training or test data.
#     - numeric_features: List of names of numeric features to scale.
#     - categorical_features: List of names of categorical features to encode.
#     - model_dir: Directory path where the preprocessing objects (encoder and
#         scaler) will be saved.
#     Returns:
#     - processed_data: A numpy array of processed features ready for training
#         or prediction.
#     """
#     # Initialize the encoder and scaler
#     encoder = OneHotEncoder(handle_unknown='ignore')
#     encoder.fit(unprocessed_data[categorical_features])
#     scaler = StandardScaler().fit(unprocessed_data[numeric_features])
#     # Transform the data
#     unprocessed_data_encoded = encoder.transform(
#         unprocessed_data[categorical_features]).toarray()
#     unprocessed_data_scaled = scaler.transform(
#         unprocessed_data[numeric_features])
#     # Combine encoded and scaled features
#     processed_data = np.hstack((
#         unprocessed_data_scaled, unprocessed_data_encoded))
#     # Save the preprocessing objects for later use
#     save_preprocessing_objects(encoder, scaler, model_dir)
#     return processed_data


# def preprocess_features_test(
#     test_data: pd.DataFrame,
#     top_10_numerical_features,
#     top_5_categorical_features,
#     model_dir: str
# ) -> np.ndarray:

#     """
#     Preprocesses the given test DataFrame by encoding categorical features
#       and scaling numerical features.
#     Args:
#     - test_data: DataFrame containing the training or test data.
#     - model_dir: Directory path where the preprocessing objects (encoder and
#         scaler) already saved.
#     Returns:
#     - test_data_processed: A numpy array of processed features ready for
#         training or prediction.
#     """
#     # location of  the encoder and scaler objects
#     encoder_path = os.path.join(model_dir, 'encoder.joblib')
#     scaler_path = os.path.join(model_dir, 'scaler.joblib')
#     # Load the encoder objects
#     loaded_encoder = joblib.load(encoder_path)
#     test_data_encoded = loaded_encoder.transform(
#         test_data[top_5_categorical_features])
#     # Load the scaler objects
#     loaded_scaler = joblib.load(scaler_path)
#     test_data_scaled = loaded_scaler.transform(
#         test_data[top_10_numerical_features])
#     # Combine scaled numeric features and encoded categorical features
#     test_data_processed = np.hstack((test_data_scaled,
#                                      test_data_encoded.toarray()))
#     return test_data_processed

def preprocess_features(data, numeric_features, categorical_features,
                        model_dir, mode='train'):
    """
    Preprocess data for training or testing.

    Args:
    - data: DataFrame with data.
    - numeric_features: List of numeric feature names to scale.
    - categorical_features: List of categorical feature names to encode.
    - model_dir: Directory path for saving/loading preprocess objects.
    - mode: 'train' for training mode, 'test' for testing mode.

    Returns:
    - A numpy array of processed features.
    """
    if mode == 'train':
        # Initialize and fit the encoder and scaler for training data
        encoder = OneHotEncoder(handle_unknown='ignore')
        scaler = StandardScaler()

        encoder.fit(data[categorical_features])
        scaler.fit(data[numeric_features])

        # Save the preprocessing objects
        save_preprocessing_objects(encoder, scaler, model_dir)
    else:
        # Load the preprocessing objects for test data
        encoder = joblib.load(os.path.join(model_dir, 'encoder.joblib'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

    # Transform the data
    data_encoded = encoder.transform(data[categorical_features]).toarray()
    data_scaled = scaler.transform(data[numeric_features])

    # Combine encoded and scaled features
    processed_data = np.hstack((data_scaled, data_encoded))

    return processed_data
