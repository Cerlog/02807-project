from pathlib import Path
import random
import pandas as pd

DATA_DIR_RAW = Path("data/raw")
DATA_DIR_PROC = Path("data/processed")


# Define file paths
USERS   = DATA_DIR_RAW / "user.json"
REVIEWS = DATA_DIR_RAW / "review.json"
REVIEWS_CLEAN = DATA_DIR_RAW / "review_clean.ndjson"
BIZ     = DATA_DIR_RAW / "business.json"
# categories of restaurants
FOOD = DATA_DIR_RAW / "Food.txt"
RESTAURANTS = DATA_DIR_RAW / "restaurants.txt"

CATEGORIES_IN_ORDER = None

def get_categories(restaurants_file, food_file):
    cats = []
    with open(restaurants_file) as f:
        for line in f:
            s = line.strip()
            if s and s != "Restaurants":
                cats.append(s)
    with open(food_file) as f:
        for line in f:
            s = line.strip()
            if s and s != "Food":
                cats.append(s)
    return cats

import hashlib
def simplify_random(categories_string):
    """
    Random version: instead of picking the first match,
    find ALL matches and pick one randomly
    """
    
    business_cats = {c.strip() for c in categories_string.split(",")}
    
    # Build a list of ALL categories that match
    matching_categories = []
    for cat in CATEGORIES_IN_ORDER:
        if cat in business_cats:
            matching_categories.append(cat)
    
    # If we found any matches, pick one at random
    if matching_categories:

        seed = int(hashlib.md5(categories_string.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return rng.choice(matching_categories)
    return "Other"


def simple_experiment(df, n_runs=100, random_seed=1337, min_threshold=0.005):
    
    distributions = []
    for i in range(n_runs):
        random.seed(random_seed + i)
        simple_cat = df["categories"].apply(simplify_random)
        dist = simple_cat.value_counts(normalize=True)
        dist.name = f"trial_{i}"
        distributions.append(dist)
        
    dist_df = pd.concat(distributions, axis=1).fillna(0)  
    # per-category stats
    summary = pd.DataFrame({
        "mean": dist_df.mean(axis=1),
        "std":  dist_df.std(axis=1),
    })
    
    summary["cv"] = summary["std"] / summary["mean"].replace(0, pd.NA)
    relevant_stats = summary[summary["mean"] >= min_threshold]
    avg_std = summary["std"].mean()
    mean_ = summary["mean"].mean()
    mean_cv = relevant_stats["cv"].mean()


    print(f"Ran {n_runs} random trials.")
    print(f"Average change in category (mean std):   {avg_std:.4f}")
    print(f"Average category share (mean of means):        {mean_:.4f}")
    print(f"Average coefficient of variation (mean cv):    {mean_cv:.4f}")
    
    # most unstable categories 
    most_unstable = relevant_stats.sort_values("cv", ascending=False).head(10)
    print("\nMost unstable categories (by CV):")
    print(most_unstable)