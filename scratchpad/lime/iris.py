# %%
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
# %%
iris = load_iris()
y = pd.DataFrame(dict(label=iris.target))
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.8)

print(iris.DESCR)
plt.hist(y["label"])
X.describe().transpose()

# %%
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train["label"])
y_train_hat = rfc.predict(X_train)

confusion_matrix(y_train_hat, y_train["label"])
y_test_hat = rfc.predict(X_test)
confusion_matrix(y_test_hat, y_test["label"])

# %%
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.to_numpy(),
    feature_names=X.columns,
    verbose=True,
    mode="classification",
    # kernel_width=2,
)

# %%
x = X.sample(1, random_state=42).to_numpy()[0]
exp = explainer.explain_instance(data_row=x, predict_fn=rfc.predict_proba)

# %%
exp.show_in_notebook(show_table=True)
exp.as_list()
rfc.predict_proba(x.reshape(1, -1))

import numpy as np
names = X.columns.to_list() + ["intercept"]
values = np.concatenate((x, np.array([1]))) * np.concatenate((lr.coef_, np.array([lr.intercept_])))
pd.DataFrame(dict(coef_name=names, coef_value=values))
sum(x * lr.coef_) + lr.intercept_

