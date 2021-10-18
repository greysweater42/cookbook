import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path("data")


def main():
    data = pd.read_csv("posts_reddit.csv")
    data["body"] = data["body"].str.replace("dog", "pet", case=False)
    data["body"] = data["body"].str.replace("Python", "pet", case=False)
    data["stage"] = np.random.choice(["train", "test"], len(data), p=[0.8, 0.2])

    nums = dict(train=defaultdict(lambda: 1), test=defaultdict(lambda: 1))
    for _, (body, label, stage) in data.iterrows():
        path = DATA_PATH / stage / label
        path.mkdir(exist_ok=True, parents=True)
        with open(path / f"{nums[stage][label]}.txt", "w") as f:
            f.write(body)
        nums[stage][label] += 1


if __name__ == "__main__":
    main()
