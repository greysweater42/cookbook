import pandas as pd

df = pd.read_csv("http://bit.ly/kaggletrain")
df.shape
df.columns

df.isna().sum()

df = df.loc[df.Embarked.notna(), ["Survived", "Pclass", "Sex", "Embarked"]]
df.shape

df.head()

X = df.loc[:, ["Pclass"]]
y = df.Survived
X.shape, y.shape

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver="lbfgs")
from sklearn.model_selection import cross_val_score

cross_val_score(logreg, X, y, cv=5, scoring="accuracy").mean()
y.value_counts(normalize=True)

df.head()
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
ohe.fit_transform(df[["Embarked"]])
ohe.categories_

X = df.drop("Survived", axis=1)
X.head()
from sklearn.compose import make_column_transformer

column_trans = make_column_transformer(
    (OneHotEncoder(), ["Sex", "Embarked"]), remainder="passthrough"
)
column_trans.fit_transform(X)

from sklearn.pipeline import make_pipeline

pipe = make_pipeline(column_trans, logreg)
cross_val_score(pipe, X, y, cv=5, scoring="accuracy").mean()