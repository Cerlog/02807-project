from pathlib import Path
import random
import pandas as pd
import hashlib
from typing import List, Optional, Union

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

def get_categories(restaurants_file: Union[str, Path], food_file: Union[str, Path]) -> List[str]:
    """
    Load restaurant and food categories from files.

    Args:
        restaurants_file: Path to the restaurants categories file.
        food_file: Path to the food categories file.

    Returns:
        List of category names.
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

def simplify_random(categories_string: str, categories_in_order: List[str], run_index: Optional[int] = None) -> str:
    """
    Randomly select a single category from a comma-separated string of categories,
    prioritizing those present in `categories_in_order`.

    Args:
        categories_string: Comma-separated string of categories.
        categories_in_order: List of valid categories to choose from.
        run_index: Optional index to seed the random number generator for reproducibility.

    Returns:
        Selected category name, or "Other" if no match found.
    """
    
    business_cats = {c.strip() for c in categories_string.split(",")}
    
    # Build a list of ALL categories that match
    matching_categories = []
    for cat in categories_in_order:
        if cat in business_cats:
            matching_categories.append(cat)
    
    # If we found any matches, pick one at random
    if matching_categories:
        if run_index is not None:
            key = categories_string
        else: 
            key = f"{categories_string}_{run_index}"

        seed = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return rng.choice(matching_categories)
    return "Other"


def test_distribution(df: pd.DataFrame, categories_in_order: List[str], n_runs: int = 100, run_index: int = 0) -> None:
    """
    Test the distribution of simplified categories over multiple random runs.

    Args:
        df: DataFrame containing a "categories" column.
        categories_in_order: List of valid categories.
        n_runs: Number of simulation runs.
        run_index: Starting seed index.
    """
    
    distributions = []
    for i in range(n_runs):
        random.seed(run_index + i)
        simple_cat = df["categories"].apply(lambda s: simplify_random(s, categories_in_order, run_index=run_index + i))
        dist = simple_cat.value_counts(normalize=True)
        dist.name = f"trial_{i}"
        distributions.append(dist)
        
    dist_df = pd.concat(distributions, axis=1).fillna(0)  
    # per-category stats
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