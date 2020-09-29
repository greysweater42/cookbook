import numpy as np
import xgboost as xgb


def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)

def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    y = dtrain.get_label()
    return ((-np.log1p(predt) + np.log1p(y) + 1) /
            np.power(predt + 1, 2))

def squared_log(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.
    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess

xgb.train({'tree_method': 'hist', 'seed': 1994},  # any other tree method is fine.
           dtrain=dtrain,
           num_boost_round=10,
           obj=squared_log)


from scipy.stats import lognorm, norm, gamma, gaussian_kde
import scipy.stats as stats
import matplotlib.pyplot as plt

y_lognorm = lognorm.rvs(s=0.5, size=1000)

m, x, _ = plt.hist(y_lognorm, density=True, bins=50)
density = gaussian_kde(y_lognorm)
plt.plot(x, density(x))

y_gamma = gamma.rvs(a=3, size=1000)
m, x, _ = plt.hist(y_gamma, density=True, bins=50)
density = gaussian_kde(y_gamma)
plt.plot(x, density(x))

e = norm.rvs(size=1000) / 10

import xgboost as xgb

from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit()
