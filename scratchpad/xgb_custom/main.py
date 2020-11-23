import random
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv'
df = pd.read_csv(url)

df = df[["Survived", "Sex", "Pclass", "Age"]].dropna()
y = df["Survived"]
X = df[["Sex", "Pclass", "Age"]]
X["Sex"] = np.where(X["Sex"] == "male", 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
sum(y_test) / len(y_test)

# decision tree
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
accuracy_score(dtc.predict(X_test), y_test)

# random forest
forest = []
k = round(len(X_train) * 0.5)
n = len(X_train)
for i in range(200):
    columns = [True] + random.choices([True, False], k=2)
    random.shuffle(columns)
    rows = random.sample(range(n), k=k)
    X_rf = X_train.iloc[rows, columns]
    y_rf = y_train.iloc[rows]
    dtc = DecisionTreeClassifier()
    dtc.fit(X_rf, y_rf)
    forest.append((dtc, columns))

votes = np.array([dtc.predict(X_test.iloc[:,columns]) for dtc, columns in forest])
y_hat = np.mean(votes, axis=0)
accuracy_score(y_hat > 0.5, y_test)

# gradient boosting
