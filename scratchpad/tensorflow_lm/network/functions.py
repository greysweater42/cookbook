from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import scale
import numpy as np
import logging
import sklearn.linear_model

logger = logging.getLogger(__name__)


def get_data():
    logging.info("started getting data")
    housing = fetch_california_housing()
    m, n = housing.data.shape
    scaled_housing_data = scale(housing.data, axis=0, with_mean=True,
                                with_std=True, copy=True)
    X_np = np.c_[np.ones((m, 1)), scaled_housing_data]
    y_np = housing.target.reshape(-1, 1)
    logging.info("finished getting data: X: {}, y: {}"
                 .format(X_np.shape, y_np.shape))
    return X_np, y_np


def split_data(X_np, y_np, batch_size, m):
    logger.info("started splitting data")
    n_batches = int(np.ceil(m / batch_size))
    X_split = np.array_split(X_np, n_batches)
    y_split = np.array_split(y_np, n_batches)
    logger.info("ended splitting data, number of batches: {}"
                .format(n_batches))
    return X_split, y_split, n_batches


def lm_sklearn(X, y):
    ls = sklearn.linear_model.LinearRegression()
    ls.fit(X, y)
    print(ls.intercept_)
    print(ls.coef_)
