import pandas as pd
import numpy as np

# Universal variables
data_folder = 'data'

def feature_summary(df):
    for column in df.columns:
        name = column
        min_value = df[column].min()
        max_value = df[column].max()
        mean = np.mean(df[column])
        median = np.median(df[column])
        sd = np.std(df[column])
        dtype = df[column].dtype

        print(f'name: {name}| mean: {mean:.2f}| med: {median:.2f}| sd: {sd:.2f}')

        if dtype != 'float64' or min_value != 0 or max_value != 1:
            extra_info = []
            if dtype != 'float64':
                extra_info.append(f'dtype: {dtype}')
            if min_value != 0:
                extra_info.append(f'min: {min_value:.2f}')
            if max_value != 1:
                extra_info.append(f'max: {max_value:.2f}')

            print(' | '.join(extra_info))

def convert_columns_to_float64(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif df[column].dtype in ['float64', 'int64', 'uint8']:
            df[column] = df[column].astype('float64')
    return df