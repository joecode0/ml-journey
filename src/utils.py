import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Universal variables
data_folder = 'data'

def find_highly_correlated_features(df, threshold=0.7):
    # Calculate the Spearman's rho correlation matrix
    corr_matrix, _ = spearmanr(df)

    # Create a DataFrame for easier indexing
    corr_df = pd.DataFrame(corr_matrix, columns=df.columns, index=df.columns)

    # Find feature pairs with correlation higher than the threshold
    correlated_pairs = []
    for i, col1 in enumerate(df.columns):
        for j, col2 in enumerate(df.columns):
            if i < j and abs(corr_df.loc[col1, col2]) > threshold:
                correlated_pairs.append((col1, col2, corr_df.loc[col1, col2]))

    # Sort the correlated pairs by absolute correlation coefficient value
    correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    # Print out the correlated feature pairs
    for col1, col2, corr in correlated_pairs:
        print(f"{col1} | {col2}: {corr:.2f}")

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