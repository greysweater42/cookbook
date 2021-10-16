import pandas as pd

data = pd.read_csv("posts_reddit.csv")
data["body"] = data["body"].str.replace("dog", "pet", case=False)
data["body"] = data["body"].str.replace("Python", "pet", case=False)
data.to_csv("data.csv", index=False)
