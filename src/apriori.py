from itertools import combinations
import pandas as pd

def baskets_parquet_to_transactions(path_parquet):
    """Load parquet with an 'items' column and return list of itemsets (as sets)."""
    b = pd.read_parquet(path_parquet)
    return [set(items) for items in b["items"]]

def baskets_df_to_transactions(df):
    return [set(items) for items in df["items"]]

def decode_itemset(itemset, id2cat_map):
    """Convert set/tuple of IDs to sorted tuple of category names."""
    if isinstance(itemset, (frozenset, set)):
        itemset = sorted(itemset)
    return tuple(id2cat_map.get(x, f"<ID_{x}>") for x in itemset)


def run_apriori(label, path_parquet, min_support, min_conf, OUT_DIR, id2cat_map):
    print("Running Apriori for", label)
    
    # load the transactions
    transactions = baskets_parquet_to_transactions(path_parquet)
    
    # run apriori 
    frequent_sets, support_map = apriori_triangular(transactions, min_support)
    rules = generate_rules(frequent_sets, transactions, min_conf)
    
    print(f"Generated {len(rules)} {label} rules")

    # 3) Turn into DataFrame and decode
    df = pd.DataFrame(rules).sort_values("Lift", ascending=False)
    df["Antecedent_decoded"] = df["Antecedent"].apply(lambda x: decode_itemset(x, id2cat_map))
    df["Consequent_decoded"] = df["Consequent"].apply(lambda x: decode_itemset(x, id2cat_map))

    # 4) Save parquet + human-readable CSV
    parquet_out = OUT_DIR / f"rules_{label}.parquet"
    csv_out     = OUT_DIR / f"rules_{label}_human.csv"

    df.to_parquet(parquet_out, index=False)
    df[["Antecedent_decoded", "Consequent_decoded", "Support", "Confidence", "Lift"]].to_csv(csv_out, index=False)

    print(f"Saved {len(df)} {label} rules")

    return df, transactions

#Triangular matrix
def pair_index(i, j, n):
    """Instead of us having a 2D matrix we save all possible pairs this way"""
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
    Apriori algorithm using triangular matrix optimization
    Input: List of transactions and minimum support 
    Output: Frequent itemsets with support counts
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
    N = len(transactions)
    cnt = sum(1 for t in transactions if itemset.issubset(t))
    return cnt / N

def generate_rules(frequent_sets, transactions, min_conf=0.4):
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
    if stars >= liked_thresh:
        return "liked"
    elif stars < hated_thresh:
        return "hated"
    return "neutral"