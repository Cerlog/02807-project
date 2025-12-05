from pathlib import Path
import random
import pandas as pd
import hashlib

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
    """
    Load restaurant and food categories from text files.

    Args:
        restaurants_file (str or Path): Path to the restaurants categories file.
        food_file (str or Path): Path to the food categories file.

    Returns:
        list: A combined list of category names found in both files.
    """
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

def simplify_random(categories_string, run_index=None):
    """
    Randomly select a single category from a comma-separated string of categories,
    prioritizing those present in the global CATEGORIES_IN_ORDER list.

    Args:
        categories_string (str): Comma-separated string of categories.
        run_index (int, optional): Seed modifier for reproducibility.

    Returns:
        str: The selected category name, or "Other" if no match is found.
    """
    
    business_cats = {c.strip() for c in categories_string.split(",")}
    
    # Build a list of ALL categories that match
    matching_categories = []
    for cat in CATEGORIES_IN_ORDER:
        if cat in business_cats:
            matching_categories.append(cat)
    
    # If we found any matches, pick one at random
    if matching_categories:
        if run_index is not None:
            key = categories_string
        else: 
            key = f"{categories_string}_{run_index}"
        # chatgpt suggestion to get reproducible random choice
        seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return rng.choice(matching_categories)
    return "Other"


def test_distribution(df, n_runs=100, run_index=0):
    """
    Test the stability of the random category simplification by running it multiple times.

    Args:
        df (pd.DataFrame): DataFrame containing a "categories" column.
        n_runs (int): Number of simulation runs.
        run_index (int): Base seed for random number generation.

    Returns:
        None: Prints summary statistics (mean, std, range) for the distribution of categories.
    """    
    
    distributions = []
    for i in range(n_runs):
        random.seed(run_index + i)
        simple_cat = df["categories"].apply(lambda s: simplify_random(s, run_index=run_index + i))
        dist = simple_cat.value_counts(normalize=True)
        dist.name = f"trial_{i}"
        distributions.append(dist)
        
    dist_df = pd.concat(distributions, axis=1).fillna(0)  
    # statistics per category 
    summary = pd.DataFrame({
        "mean": dist_df.mean(axis=1),
        "std":  dist_df.std(axis=1),
    })
    
    for col in ["mean", "std"]:
        col_data = summary[col]
        print(f"{col}:")
        print(f"  Mean: {col_data.mean():.2e}")
        print(f"  Std Dev: {col_data.std():.2e}")
        print(f"  Range: [{col_data.min():.2e}, {col_data.max():.2e}]")
        print()
        
    summary.to_csv(DATA_DIR_PROC / "simple_experiment_summary.csv")