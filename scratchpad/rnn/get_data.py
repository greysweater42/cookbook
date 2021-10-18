import praw
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from psaw import PushshiftAPI


# reading reddit credentials
reddit_creds_path = Path.home() / "cookbook" / "scratchpad" / "rnn" / "creds.json"
with open(reddit_creds_path, "r") as f:
    reddit_creds = json.load(f)

# setting up a connection
reddit = praw.Reddit(
    client_id=reddit_creds["client_id"],
    client_secret=reddit_creds["client_secret"],
    user_agent="python_or_dog",
    username=reddit_creds["username"],
    password=reddit_creds["password"],
)
api = PushshiftAPI(reddit)

# a function for downloading posts
def get_data(api, subreddit_name):
    submissions = api.search_submissions(limit=10000, subreddit=subreddit_name)
    bodies = []
    for submission in tqdm(submissions, total=10000):
        bodies.append(submission.selftext)
    topics_data = pd.DataFrame(dict(body=bodies, label=subreddit_name))
    return topics_data


# downloading posts
all_posts = dict()
for pet in ["dogs", "Python"]:
    raw_data = get_data(api, subreddit_name=pet)
    all_posts[pet] = raw_data[raw_data["body"].str.len() > 200]
    print("downloading {} finished".format(pet))
    del raw_data  # save some memory

# saving results
result = pd.concat(all_posts.values())
save_path = Path.home() / "cookbook" / "scratchpad" / "rnn"
result.to_csv(save_path / "posts_reddit.csv", index=False)
