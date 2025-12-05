from itertools import combinations
import pandas as pd

def baskets_parquet_to_transactions(path_parquet):
    """
    Load parquet with an 'items' column and return list of itemsets (as sets).
    Args:
        path_parquet (str or Path): Path to the parquet file.
    Returns:
        list: A list of sets, where each set represents a transaction.
    """
    b = pd.read_parquet(path_parquet)
    return [set(items) for items in b["items"]]

def baskets_df_to_transactions(df):
    """
    Convert a DataFrame with an 'items' column into a list of transaction sets.

    Args:
        df (pd.DataFrame): DataFrame containing an 'items' column.

    Returns:
        list: A list of sets, where each set represents a transaction.
    """
    return [set(items) for items in df["items"]]

def decode_itemset(itemset, id2cat_map):
    """
    Convert set/tuple of IDs to sorted tuple of category names.
    Args:
        itemset (set or tuple): Set or tuple of item IDs.
        id2cat_map (dict): Mapping from item ID to category name.
    Returns:
        tuple: Sorted tuple of category names corresponding to the item IDs.
    """
    if isinstance(itemset, (frozenset, set)):
        itemset = sorted(itemset)
    return tuple(id2cat_map.get(x, f"<ID_{x}>") for x in itemset)


def run_apriori(label, path_parquet, min_support, min_conf, OUT_DIR, id2cat_map):
    """
    Run Apriori algorithm on transactions from parquet file and save results.
    Args:
        label (str): Label for the dataset (e.g., 'overall', 'community_1').
        path_parquet (str or Path): Path to the parquet file with transactions.
        min_support (float): Minimum support threshold.
        min_conf (float): Minimum confidence threshold.
        OUT_DIR (str or Path): Directory to save output files.
        id2cat_map (dict): Mapping from item ID to category name.
    Returns:
        pd.DataFrame: DataFrame containing the generated association rules.
        list: List of transactions used for the Apriori algorithm.
    """
    print("Running Apriori for", label)
    
    # load the transactions
    transactions = baskets_parquet_to_transactions(path_parquet)
    
    # run apriori 
    frequent_sets, support_map = apriori_triangular(transactions, min_support)
    rules = generate_rules(frequent_sets, transactions, min_conf)
    
    print(f"Generated {len(rules)} {label} rules")

    # Turn into DataFrame and decode
    df = pd.DataFrame(rules).sort_values("Lift", ascending=False)
    df["Antecedent_decoded"] = df["Antecedent"].apply(lambda x: decode_itemset(x, id2cat_map))
    df["Consequent_decoded"] = df["Consequent"].apply(lambda x: decode_itemset(x, id2cat_map))

    # Save parquet and readable CSV
    parquet_out = OUT_DIR / f"rules_{label}.parquet"
    csv_out     = OUT_DIR / f"rules_{label}_human.csv"

    df.to_parquet(parquet_out, index=False)
    df[["Antecedent_decoded", "Consequent_decoded", "Support", "Confidence", "Lift"]].to_csv(csv_out, index=False)

    print(f"Saved {len(df)} {label} rules")

    return df, transactions

#Triangular matrix
def pair_index(i, j, n):
    """
    Calculate the linear index for a pair (i, j) in a triangular matrix.
    
    This avoids storing a full 2D matrix by mapping pairs to a 1D array index.
    
    Args:
        i (int): Row index.
        j (int): Column index.
        n (int): Dimension of the matrix (number of items).

    Returns:
        int: The linear index corresponding to the pair (i, j).
    """
    #Only count pair once
    if i > j:
        i, j = j, i
    return int(i * (n - (i + 1) / 2) + (j - i - 1))

#Support function
def support(itemset, transactions):
    """Calculate support for an itemset in the list of transactions"""
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)

def apriori_triangular(transactions, min_support):
    """
    Execute the Apriori algorithm using a triangular matrix optimization for 2-itemsets.

    Args:
        transactions (list): A list of sets, where each set represents a transaction.
        min_support (float): The minimum support threshold.

    Returns:
        tuple: A tuple containing:
            - all_frequent (list): A list of lists, where each inner list contains frequent itemsets of size k.
            - support_map (dict): A dictionary mapping frequent itemsets to their support values.
    """
    '''
    #1. Individual item counts
    Sort items by alphabetical order, indexing
    '''
    #Sort unique items by alphabetical order
    items = sorted(set(i for t in transactions for i in t))
    len_items = len(items)
    #Index items
    index = {item: idx for idx, item in enumerate(items)}
 
    #Count single items
    single_counts = [0] * len_items
    for t in transactions:
        for item in t:
            single_counts[index[item]] += 1

    #Get frequent 1-itemsets
    N = len(transactions)
    min_count = min_support * N
    frequent_items = [items[i] for i, count in enumerate(single_counts) if count >= min_count]
    L1 = [frozenset([item]) for item in frequent_items]
    
    
        # support dict (store supports for all frequent sets)
    support_map = {}
    for item in frequent_items:
        support_map[frozenset([item])] = single_counts[index[item]] / N
    
    
    '''
    #2. Pair counts using triangular matrix
    Count how many times each pair of frequent items appear together
    '''
    pair_len = len(frequent_items)
    #All possible pairs for item
    
    # map item -> index *insidefrequent items
    freq_index = {item: idx for idx, item in enumerate(frequent_items)}
    
    pair_counts = [0] * (pair_len * (pair_len - 1) // 2)

    for t in transactions:
        #Only frequent items in transaction
        trans_idx = sorted(freq_index[i] for i in t if i in freq_index)
        
        for a in range(len(trans_idx)):
            for b in range(a + 1, len(trans_idx)):
                i = trans_idx[a]
                j = trans_idx[b]
                k = pair_index(i, j, pair_len)
                pair_counts[k] += 1

    #Get frequent 2-itemsets
    L2= list()
    for i in range(pair_len):
        for j in range(i + 1, pair_len):
            k = pair_index(i,j, pair_len)
            if pair_counts[k] >= min_count:
                L2.append(frozenset([frequent_items[i], frequent_items[j]]))

    '''
    #3. Generate larger itemsets (k>=3)
    Using previous frequent itemsets to generate candidates
    '''
    #Previous frequent itemsets
    all_frequent = [L1, L2]
    k = 3
    while True:
        #Until no more frequent itemsets
        prev_frequent = all_frequent[-1]
        if not prev_frequent:
            break
    
        candidates = set()
        prev_list = list(prev_frequent)
        len_prev = len(prev_list)
        #Iterate over previous
        for i in range(len_prev):
            for j in range(i + 1, len_prev):
                #Join items until we have k items
                #Generate union of two itemsets present in L2
                union = prev_list[i].union(prev_list[j])
                if len(union) == k:
                    #All possible combinations of size k-1 must be frequent
                    subsets = combinations(union, k-1)
                    if all(frozenset(s) in prev_frequent for s in subsets):
                        #If frequent, add to candidates
                        candidates.add(frozenset(union))

        # Count support and keep those above threshold
        next_level = [c for c in candidates if support(c, transactions) >= min_support]
        if not next_level:
            break

        all_frequent.append(next_level)
        k += 1
    return all_frequent, support_map


def support_of(itemset, transactions):
    """
    Calculate the support of a specific itemset.

    Args:
        itemset (set or frozenset): The itemset to check.
        transactions (list): List of transactions.

    Returns:
        float: Support value.
    """
    N = len(transactions)
    cnt = sum(1 for t in transactions if itemset.issubset(t))
    return cnt / N

def generate_rules(frequent_sets, transactions, min_conf=0.4):
    """
    Generate association rules from frequent itemsets.

    Args:
        frequent_sets (list): List of frequent itemsets (output from apriori).
        transactions (list): List of transactions.
        min_conf (float): Minimum confidence threshold.

    Returns:
        list: A list of dictionaries, each representing a rule with metrics (Support, Confidence, Lift).
    """    
    # collect supports for all frequent itemsets (including L1)
    sup = {}
    for Lk in frequent_sets:
        for S in Lk:
            if S not in sup:
                sup[S] = support_of(S, transactions)

    rules = []
    for Lk in frequent_sets:
        for S in Lk:
            if len(S) < 2:
                continue
            # all non-empty proper antecedents
            for rlen in range(1, len(S)):
                for A in map(frozenset, combinations(S, rlen)):
                    B = S - A
                    if not B:
                        continue
                    conf = sup[S] / max(sup[A], 1e-12)
                    if conf >= min_conf:
                        lift = conf / max(sup[B], 1e-12)
                        rules.append({
                            "Antecedent": tuple(sorted(A)),
                            "Consequent": tuple(sorted(B)),
                            "Support": sup[S],
                            "Confidence": conf,
                            "Lift": lift
                        })
    return rules


def filter_rules(df_rules, min_support=0.015, min_confidence=0.60, 
                         min_lift=1.5, max_antecedent=3, max_consequent=3, save_path=None, color="cm.Greens"):
    """
    Filter and style association rules DataFrame based on given thresholds.
    Args:
        df_rules: DataFrame containing association rules.
        min_support: Minimum support threshold.
        min_confidence: Minimum confidence threshold.
        min_lift: Minimum lift threshold.
        max_antecedent: Maximum size of antecedent.
        max_consequent: Maximum size of consequent.
        save_path: Optional path to save the styled HTML.
        color: Color map for styling.
    """
    rules_filtered = df_rules[
        (df_rules["Support"] > min_support) &
        (df_rules["Confidence"] > min_confidence) &
        (df_rules["Lift"] > min_lift) &
        (df_rules["Consequent"].apply(len) <= max_consequent) &
        (df_rules["Antecedent"].apply(len) <= max_antecedent)
    ]
    
    num_cols = ["Support", "Confidence", "Lift"]
    styled = rules_filtered.style \
        .format({c: "{:.4f}" for c in num_cols}) \
        .background_gradient(cmap=color, subset=num_cols)
    
    #if save_path:
    #    styled.to_html(save_path)
    
    return rules_filtered


def filter_baskets_by_community(baskets_df, user_ids_set):
    """Filter baskets to only include users in the given set"""
    filtered = baskets_df[baskets_df['user_id'].isin(user_ids_set)].copy()
    return filtered

def jaccard_similariy(df1, df2, ant_col="Antecedent", cons_col="Consequent"):
    """
    Calculate the Jaccard similarity between the rules of two DataFrames.

    Args:
        df1 (pd.DataFrame): First DataFrame of rules.
        df2 (pd.DataFrame): Second DataFrame of rules.
        ant_col (str): Column name for antecedents.
        cons_col (str): Column name for consequents.

    Returns:
        float: Jaccard similarity score.
    """
    
    # making antecendent and consequent tuples 
    ants1 = df1[ant_col].apply(tuple)
    cons1 = df1[cons_col].apply(tuple)
    ants2 = df2[ant_col].apply(tuple)
    cons2 = df2[cons_col].apply(tuple)

    # create set of rules     
    set1 = set(zip(ants1, cons1))
    set2 = set(zip(ants2, cons2))
    
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    if (len(union) == 0):
        jaccard_similarity_value = 1.0 if len(intersection) == 0 else 0.0
    else:
        jaccard_similarity_value = len(intersection) / len(union)
    
    return jaccard_similarity_value

def label_yelp_liked(stars, liked_thresh=4.0, hated_thresh=3.0):
    """
    Label a review as 'liked', 'hated', or 'neutral' based on star rating.

    Args:
        stars (float): Star rating.
        liked_thresh (float): Threshold for 'liked'.
        hated_thresh (float): Threshold for 'hated'.

    Returns:
        str: Label ('liked', 'hated', 'neutral').
    """    
    if stars >= liked_thresh:
        return "liked"
    elif stars < hated_thresh:
        return "hated"
    return "neutral"