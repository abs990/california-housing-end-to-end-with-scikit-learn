"""
California Housing, End to End with Scikit-Learn

Assembled from your step-by-step solutions.
"""

import os
import numpy as np
import tempfile
import tarfile
import urllib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib


# Step 1 - load_housing
def load_housing():
    path = os.path.join(tempfile.gettempdir(), "housing.tgz")

    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://github.com/ageron/data/raw/main/housing.tgz", path
        )

    with tarfile.open(path, "r") as tar:
        f = tar.extractfile("housing/housing.csv")
        return pd.read_csv(f)


# Step 2 - income_categories
def income_categories(df):
    # pd.cut median_income with edges [0, 1.5, 3, 4.5, 6, inf] and labels 1..5; return an int Series.
    return pd.cut(
        df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    ).astype(int)


# Step 3 - stratified_split
def stratified_split(df, test_size=0.2, random_state=42):
    # train_test_split stratified on income_categories(df); return (train_set, test_set).
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
    corr_matrix = (
        corr_matrix["median_house_value"]
        .drop("median_house_value")
        .sort_values(ascending=False)
    )
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


# Step 7 - ClusterSimilarity
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        # fit KMeans(n_clusters, n_init=10, random_state) on X with sample_weight;
        # keep it as self.kmeans_
        self.kmeans_ = KMeans(
            self.n_clusters, n_init=10, random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        # rbf_kernel similarity of each row of X to the cluster centers
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        # ["Cluster 0 similarity", ...]
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


# Step 8 - numeric_pipeline
def numeric_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler())


# Step 9 - categorical_pipeline
def categorical_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="most_frequent", missing_values=None),
        OneHotEncoder(handle_unknown="ignore"),
    )


# Step 10 - build_preprocessing
def build_preprocessing(n_clusters=10, gamma=1.0, random_state=42):
    # ColumnTransformer with 'log', 'geo', 'cat' transformers and remainder=numeric_pipeline().
    log = make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    geo = ClusterSimilarity(n_clusters, gamma, random_state)
    cat = categorical_pipeline()
    remainder = numeric_pipeline()

    return ColumnTransformer(
        [
            (
                "log",
                log,
                [
                    "total_bedrooms",
                    "total_rooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            ("geo", geo, ["latitude", "longitude"]),
            ("cat", cat, ["ocean_proximity"]),
        ],
        remainder=remainder,
    )


# Step 11 - rmse
def rmse(y_true, y_pred):
    # sqrt(mean((y_true - y_pred)^2)) as a float.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# Step 12 - dummy_baseline_rmse
def dummy_baseline_rmse(X, y):
    # fit DummyRegressor(strategy='mean') and return its RMSE on (X, y).
    # Initialize the Dummy Regressor with the 'mean' strategy
    dummy_regr = DummyRegressor(strategy="mean")

    # Fit the model
    dummy_regr.fit(X, y)

    # Make predictions
    y_pred = dummy_regr.predict(X)

    return rmse(y_pred, y)


# Step 13 - cross_val_rmse
def cross_val_rmse(model, X, y, cv=3):
    # cross_val_score with neg_root_mean_squared_error; return {'scores': [...], 'mean': ..., 'std': ...}.
    scores = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)

    res = {}
    res["scores"] = scores.astype(float).tolist()
    res["mean"] = float(scores.mean())
    res["std"] = float(scores.std())

    return res


# Step 14 - linear_model
def linear_model(preprocessing):
    return make_pipeline(preprocessing, LinearRegression())


# Step 15 - forest_model
def forest_model(preprocessing, n_estimators=50, random_state=42):
    return make_pipeline(
        preprocessing,
        RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
    )


# Step 16 - random_search
def random_search(pipeline, X, y, n_iter=5, cv=3, random_state=42):
    # RandomizedSearchCV over geo n_clusters 3..10 and forest max_features 2..8; fit and return it.
    param_distribs = {
        "columntransformer__geo__n_clusters": list(range(3, 11)),
        "randomforestregressor__max_features": list(range(2, 9)),
    }

    return RandomizedSearchCV(
        pipeline,
        param_distribs,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
    ).fit(X, y)


# Step 17 - test_rmse
def test_rmse(model, test_set):
    # add ratio features, split, predict with the fitted model, return rmse.
    X_test, y_test = split_features_labels(add_ratio_features(test_set))

    return rmse(y_test, model.predict(X_test))


# Step 18 - bootstrap_rmse_ci
def bootstrap_rmse_ci(y_true, y_pred, n_boot=200, alpha=0.05, random_state=42):
    # bootstrap the RMSE by resampling index pairs;
    # return (low, high) percentiles as floats.

    rng = np.random.default_rng(random_state)
    n = len(y_true)

    boots = [0] * n_boot

    for itr in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[itr] = rmse(
            y_true.iloc[idx] if isinstance(y_true, pd.Series) else y_true[idx],
            y_pred[idx],
        )

    p_values = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(p_values[0]), float(p_values[1])


# Step 19 - feature_importances
def feature_importances(search, k=5):
    # top-k (importance, name) tuples from the best estimator,
    # importances rounded to 3 decimals.
    est = search.best_estimator_
    names = est.steps[0][1].get_feature_names_out()
    imps = est.steps[-1][1].feature_importances_

    return sorted(zip(imps.round(3), names), reverse=True)[:k]


# Step 20 - worst_errors
def worst_errors(model, df, k=5):
    # DataFrame of the k largest absolute errors
    # with columns actual, predicted, abs_error.

    X, y = split_features_labels(add_ratio_features(df))

    y_pred = model.predict(X)

    ret = pd.DataFrame(
        {"actual": y, "predicted": y_pred, "abs_error": np.abs(y - y_pred)},
        index=df.index,
    )

    ret = ret.sort_values("abs_error", ascending=False).head(k)
    return ret


# Step 21 - save_and_reload
def save_and_reload(model, path):
    # TODO: joblib.dump then joblib.load; return the reloaded model.
    joblib.dump(model, path)
    return joblib.load(path)


# Step 22 - predict_new
def predict_new(model, districts):
    # DataFrame from the dicts
    # add ratio features, predict, return floats rounded to the dollar.
    df = pd.DataFrame(districts)
    df = add_ratio_features(df)

    return [float(round(p)) for p in model.predict(df)]
