import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# Universal variables
data_folder = 'data'

def find_highly_correlated_features(df, target_col, threshold=0.7, debug=False):
    # Calculate Spearman's rho for all pairs of features, and target variable correlations
    spearman_corr = df.corr(method='spearman')
    target_corr = spearman_corr[target_col]

    # Create a list to store the relevant information
    correlations = []

    # Loop through the correlation matrix and collect the relevant information
    for i in range(len(spearman_corr)):
        for j in range(i + 1, len(spearman_corr)):
            if abs(spearman_corr.iloc[i, j]) > threshold:
                feature1 = spearman_corr.columns[i]
                feature2 = spearman_corr.columns[j]
                if feature1 == target_col or feature2 == target_col:
                    continue
                corr = spearman_corr.iloc[i, j]
                var1 = df[feature1].var()
                var2 = df[feature2].var()
                target_corr1 = target_corr[feature1]
                target_corr2 = target_corr[feature2]
                correlations.append((feature1, feature2, corr, var1, var2, target_corr1, target_corr2))

    # Sort the list by absolute correlation coefficient
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    # Print the sorted correlations
    if debug:
        for corr in correlations:
            print(f"({corr[0]}, {corr[1]}): Rho:{corr[2]:.2f} || Var:{corr[3]:.2f}|{corr[4]:.2f} || Target cor:{corr[5]:.2f}|{corr[6]:.2f}")

    return correlations

def select_features_to_remove(correlations, debug=False):
    # Select the features to remove based on correlation with target variable, variance then name length
    features_to_remove = set()
    for corr in correlations:
        feature1, feature2 = corr[0], corr[1]
        if feature1 in features_to_remove or feature2 in features_to_remove:
            continue

        if corr[5] > corr[6]:
            features_to_remove.add(feature2)
        elif corr[5] < corr[6]:
            features_to_remove.add(feature1)
        else:
            if corr[3] > corr[4]:
                features_to_remove.add(feature2)
            elif corr[3] < corr[4]:
                features_to_remove.add(feature1)
            else:
                features_to_remove.add(max(feature1, feature2, key=len))

    if debug:
        print("Number of features to remove: {}".format(len(features_to_remove)))
        print("Features to remove: {}".format(features_to_remove))
    return list(features_to_remove)


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