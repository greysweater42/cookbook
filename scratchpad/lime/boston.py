# %%

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# %%

boston = load_boston()
y = pd.DataFrame(dict(price=boston.target))
X = pd.DataFrame(boston.data, columns=boston.feature_names)
print(boston.DESCR)

plt.hist(y["price"])

X.describe().transpose()

# %%
lr = LinearRegression(fit_intercept=False)
lr.fit(X, y["price"])
y_hat = lr.predict(X)

plt.scatter(y_hat, y["price"])

names = X.columns.to_list() + ["intercept"]
values = list(lr.coef_) + [lr.intercept_]
pd.DataFrame(dict(coef_name=names, coef_value=values))

# %%
import lime
import lime.lime_tabular
import numpy as np


x = X.sample(1, random_state=42).to_numpy()[0]

exps = list()
x_rand = np.arange(1, 31) / 2
for w in x_rand:
    print(w)
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.to_numpy(),
        feature_names=X.columns,
        verbose=True,
        mode="regression",
        kernel_width=w,
    )
    exp = explainer.explain_instance(
        data_row=x, predict_fn=lr.predict, num_samples=100, num_features=len(X.columns)
    )
    exps.append(exp)
# %%
exps_list = [e.as_list() for e in exps]
import matplotlib.pyplot as plt

res = {col: [] for col in X.columns}
for r in res:
    for e in exps_list:
        val = [f for f in e if r in f[0]]
        res[r].append(val[0][1] if val else 0)

for key, value in res.items():
    plt.plot(x_rand, value, label=key)

# %%
exps[10].score
exps[10].local_pred

# %%
import numpy as np

names = X.columns.to_list() + ["intercept"]
values = np.concatenate((x, np.array([1]))) * np.concatenate(
    (lr.coef_, np.array([lr.intercept_]))
)
pd.DataFrame(dict(coef_name=names, coef_value=values))
sum(x * lr.coef_) + lr.intercept_

# %%
