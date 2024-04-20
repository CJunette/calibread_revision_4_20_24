import os

import pandas as pd


def read_model_config():
    file_path = f"{os.getcwd()}/data/model_config/model_config.csv"
    model_config_df = pd.read_csv(file_path)

    return model_config_df