from .preprocess import clean_data
from .preprocess import fill_missing_values
from .preprocess import load_features
from .preprocess import preprocess_features
import joblib
import os
import numpy as np
import pandas as pd


def make_predictions(test_data: pd.DataFrame) -> np.ndarray:
    # storing test_data_ids into a list
    test_data_id = list(test_data['Id'])
    model_dir = '/home/sachin/DSP/dsp-anandhu-krishna/models'
    test_data_cleand = clean_data(test_data)
    test_data_cleand = fill_missing_values(test_data_cleand)
    # Specify the file path from where you want to load the JSON
    feature_file_path = ('/home/sachin/DSP/dsp-anandhu-krishna/'
                         'models/features_dictionary.json')
    top_10_numerical_features, top_5_categorical_features =\
        load_features(feature_file_path)
    test_data_processed = preprocess_features(
        test_data_cleand,
        top_10_numerical_features,
        top_5_categorical_features,
        model_dir,
        'test')
    model_path = os.path.join(model_dir, 'model.joblib')
    # model loading
    model = joblib.load(model_path)
    # model predicitng
    test_data_pred = model.predict(test_data_processed)
    # storing the predicted slaes price into a df
    test_data_pred_df = pd.DataFrame(
        test_data_pred, columns=['pred_sales_price'])
    # converting the Ids into a dataframe
    test_data_id_df = pd.DataFrame(test_data_id, columns=['id'])
    # Concatenate the new DataFrame with 'test_data_id_df'
    df_combined = pd.concat([test_data_id_df, test_data_pred_df], axis=1)
    return df_combined
