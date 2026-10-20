"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_housing
def load_housing():
    import os
    import tarfile
    import pandas as pd
    
    path = os.path.join(tempfile.gettempdir(), "housing.tgz")
    
    if not os.path.exists(path):
        urllib.request.urlretrieve('https://github.com/ageron/data/raw/main/housing.tgz', path)
    
    with tarfile.open(path, "r") as tar:
        f = tar.extractfile("housing/housing.csv")
        return pd.read_csv(f)

# Step 2 - income_categories
def income_categories(df):
    # pd.cut median_income with edges [0, 1.5, 3, 4.5, 6, inf] and labels 1..5; return an int Series.
    return pd.cut(df["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)

# Step 3 - stratified_split
def stratified_split(df, test_size=0.2, random_state=42):
    # train_test_split stratified on income_categories(df); return (train_set, test_set).
    from sklearn.model_selection import train_test_split
    strata = income_categories(df)
    train_set, test_set = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=strata
    )
    return train_set, test_set

# Step 4 - explore_correlations
def explore_correlations(df):
    # Pearson correlation of every numeric column with median_house_value, 
    # sorted descending, target excluded.
    corr_matrix = df.corr(numeric_only=True)
    corr_matrix = corr_matrix["median_house_value"].drop("median_house_value").sort_values(ascending=False)
    return corr_matrix

# Step 5 - add_ratio_features
def add_ratio_features(df):
    # Return a copy with rooms_per_house, bedrooms_ratio and people_per_house columns added.
    df_new = df.copy()

    df_new["rooms_per_house"] = df_new["total_rooms"] / df["households"]
    df_new["bedrooms_ratio"] = df_new["total_bedrooms"] / df_new["total_rooms"]
    df_new["people_per_house"] = df_new["population"] / df["households"]

    return df_new

# Step 6 - split_features_labels
def split_features_labels(df):
    # Return (X without median_house_value, y = median_house_value Series).
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"].copy()
    return X, y

# Step 7 - ClusterSimilarity (not yet solved)
# TODO: implement

# Step 8 - numeric_pipeline (not yet solved)
# TODO: implement

# Step 9 - categorical_pipeline (not yet solved)
# TODO: implement

# Step 10 - build_preprocessing (not yet solved)
# TODO: implement

# Step 11 - rmse (not yet solved)
# TODO: implement

# Step 12 - dummy_baseline_rmse (not yet solved)
# TODO: implement

# Step 13 - cross_val_rmse (not yet solved)
# TODO: implement

# Step 14 - linear_model (not yet solved)
# TODO: implement

# Step 15 - forest_model (not yet solved)
# TODO: implement

# Step 16 - random_search (not yet solved)
# TODO: implement

# Step 17 - test_rmse (not yet solved)
# TODO: implement

# Step 18 - bootstrap_rmse_ci (not yet solved)
# TODO: implement

# Step 19 - feature_importances (not yet solved)
# TODO: implement

# Step 20 - worst_errors (not yet solved)
# TODO: implement

# Step 21 - save_and_reload (not yet solved)
# TODO: implement

# Step 22 - predict_new (not yet solved)
# TODO: implement

