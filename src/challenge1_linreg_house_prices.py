import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder

from src.utils import *

def pipeline1(debug=False):
    
    # Read in relevant data
    train = pd.read_csv('data/house-prices/train.csv')
    if debug:
        print("Original Train Dataframe:")
        print(train.head())
        print('Original shape of Training Dataframe: {}'.format(train.shape))

    # Run preprocessing
    if debug:
        print('Preprocessing in progress...')
    train = run_preprocessing(train,"SalePrice",debug)

    # Perform grid search of range of correlation thresholds
    if debug:
        print('Grid search in progress...')
    threshold_values = np.arange(0.7, 0.75, 0.05)
    t, model, df_processed = grid_search_correlation_threshold(train, "SalePrice", threshold_values, debug)

    sys.exit(0)

    # Perform feature selection
    k_values = multiples(train.shape[1]-1,1)
    best_k = find_k_best_features(train, k_values, "SalePrice",debug)
    
    return

def grid_search_correlation_threshold(df, target_col, threshold_values, debug=False):
    best_threshold = None
    best_mse = float('inf')
    best_r2 = 0
    best_model = None
    best_processed_df = None

    for threshold in threshold_values:
        if debug:
            print(f"Processing threshold: {threshold}")

        # Find highly correlated features based on the current threshold
        correlations = find_highly_correlated_features(df, target_col, threshold)

        # Select features to remove
        features_to_remove = select_features_to_remove(correlations)

        # Remove selected features
        processed_df = df.copy().drop(features_to_remove, axis=1)

        if debug:
            print(f"Processed dataframe shape: {processed_df.shape}")
            print("Processed dataframe:")
            print(processed_df.head())

        # Evaluate the model using cross-validation
        mse, r2 = cross_validate_linear_regression(processed_df, target_col,debug=True)

        num_features_removed = len(features_to_remove)

        if debug:
            print(f"Threshold: {threshold} | MSE: {mse:.2f} | R2: {r2:.2f} | Features removed: {num_features_removed}")

        # Update the best threshold, model, and processed dataset if the current one is better
        if mse < best_mse:
            best_threshold = threshold
            best_mse = mse
            best_r2 = r2
            best_processed_df = processed_df
            best_features_removed = num_features_removed

    if debug:
        print(f"Best threshold: {best_threshold} | Best MSE: {best_mse:.2f} | Best R2: {best_r2:.2f} | Best Features removed: {best_features_removed}")

    # Train the best model on the entire processed dataset
    X_best = best_processed_df[[x for x in best_processed_df.columns if x != target_col]]
    y_best = best_processed_df[target_col]
    best_model = LinearRegression().fit(X_best, y_best)

    return best_threshold, best_model, best_processed_df

def cross_validate_linear_regression(df, target_col, num_splits=10, random_state=42, debug=False):
    # Prepare the data (assuming X and y are your feature matrix and target vector)
    X = df[[x for x in df.columns if x != target_col]]
    y = df[target_col]

    # Create a KFold cross-validator
    if debug:
        print('Generating KFold for cross-validation...')
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)

    mse_scores = []
    r2_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Remove constant columns within the split
        X_train = remove_constant_features(X_train)
        X_val = X_val[X_train.columns]  # Keep only the same columns as in X_train

        # Convert X_train and y_train to float64
        X_train = X_train.astype(np.float64)
        y_train = y_train.astype(np.float64)

        # Normalize the target variable
        y_train_normalized = (y_train - y_train.min()) / (y_train.max() - y_train.min())
        y_val_normalized = (y_val - y_train.min()) / (y_train.max() - y_train.min())

        # Train the linear regression model
        model = train_linear_regression(X_train, y_train_normalized)

        # Make predictions and evaluate the model
        y_pred_normalized = model.predict(X_val)

        # Reverse the normalization for the predictions
        y_pred = y_pred_normalized * (y_train.max() - y_train.min()) + y_train.min()

        # Evaluate the model
        mse, r2 = evaluate_model(y_val, y_pred, debug)

        # Append the scores for this split to the lists
        mse_scores.append(mse)
        r2_scores.append(r2)

    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    return avg_mse, avg_r2

def train_linear_regression(X_train, y_train):
    # Create a linear regression model
    model = LinearRegression()

    # Train the model on the training data
    model = model.fit(X_train, y_train)

    return model

def evaluate_model(y_val, y_pred, debug=False):
    if debug:
        print("First 5 predictions:")
        print(y_pred[:5])
        print("First 5 actual values:")
        print(y_val[:5])

    # Calculate the mean squared error (MSE)
    mse = mean_squared_error(y_val, y_pred)

    # Calculate the coefficient of determination (R-squared)
    r2 = r2_score(y_val, y_pred)

    return mse, r2

def run_preprocessing(df, target_col, debug=False):
    # Drop the useless features
    df = drop_useless_features(df, debug)

    # Replace categorical columns that use numerical values with the actual category names
    df = convert_numerical_category_columns(df, debug)

    # Fill in the missing values
    df = fill_in_missing_values(df, debug)

    # One-hot encode the remaining categorical features
    df = one_hot_encode_non_sparse_categorical_features(df, [], debug)
    
    # If debug, print the feature summary
    if debug:
        print("Here's the post-encoding feature summary:")
        feature_summary(df)

    # Remove constant features
    df = remove_constant_features(df, debug)

    # Remove outliers from the dataset using the IQR method
    df = remove_outliers_iqr(df, factor=5, debug=debug)

    # Print the head of the dataframe
    if debug:
        print("Here's the head of the dataframe:")
        print(df.head())
        print('And the shape: {}'.format(df.shape))

    # Identify high variance features & apply log transformation
    high_variance_columns = identify_high_variance_columns(df, threshold=1.0, debug=debug)  # Adjust threshold if needed
    df = apply_transformations(df, high_variance_columns, transformation='log', debug=debug)

    # Print the head of the dataframe
    if debug:
        print("Here's the head of the dataframe:")
        print(df.head())
        print('And the shape: {}'.format(df.shape))

    if debug:
        feature_summary(df)

    # Standardize all features
    df = standardize_features(df, [x for x in list(df.columns) if x != target_col], debug)

    # Print the head of the dataframe
    if debug:
        print("Processing complete. Here's the head of the dataframe:")
        print(df.head())
        print('And the shape: {}'.format(df.shape))

    if debug:
        feature_summary(df)

    return df

def drop_useless_features(df,debug=False):
    if debug:
        print('Dropping useless features...')
        print('Original count of columns: {}'.format(len(df.columns)))
    
    # Drop the unnecessary features
    df = df.drop(['Id'],axis=1)

    if debug:
        print('Count of columns after dropping useless features: {}'.format(len(df.columns)))
    return df

def convert_numerical_category_columns(df, debug=False):
    if debug:
        print("Converting numerical category columns to strings...")

    numerical_categorical_cols = ['MSSubClass','OverallQual','OverallCond','MoSold','YrSold']
    for col in numerical_categorical_cols:
        df[col] = df[col].astype(str)

    return df

def one_hot_encode_non_sparse_categorical_features(df, sparse_categorical_columns, debug=False):

    if debug:
        print("One-hot-encoding remaining categorical features...")

    # Get list of remaining categorical features
    remaining_categorical_features = [col for col in df.columns if df[col].dtype == 'object' and col not in sparse_categorical_columns]
    
    # One-hot-encode remaining categorical features
    df_one_hot_encoded = pd.get_dummies(df[remaining_categorical_features])
    for column_name in df_one_hot_encoded.columns:
        df_one_hot_encoded[column_name] = df_one_hot_encoded[column_name].astype(float)

    # Drop original features
    df.drop(remaining_categorical_features, axis=1, inplace=True)

    # Merge one-hot-encoded features with the rest of the data
    df = pd.concat([df, df_one_hot_encoded], axis=1)

    df = convert_columns_to_float64(df)

    if debug:
        print('Count of columns after one-hot-encoding: {}'.format(len(df.columns)))

    return df

def multiples(n,m):
    multiples = []
    for i in range(m, n, m):
        multiples.append(i)
    return multiples
