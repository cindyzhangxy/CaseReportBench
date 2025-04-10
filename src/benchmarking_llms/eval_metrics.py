import ray
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
from Levenshtein import distance
import math
import evaluate
 

ray.init(ignore_reinit_error=True, num_cpus=12)
@ray.remote
def compute_token_set_ratio(list1, lists2, method="average"):
    """
    Compute token set ratio for lists of strings with specified aggregation method.
    """
    # Handle NA scenario
    if (list1 == []) and (list2 == [] ):
        return 100  # Perfect match for both being empty 
    
    if (list1 == [] and list2 != []) or (list1 != [] and list2 == []):
        return 0  # One list is empty, assign 0

    if list1 is None or list2 is None or not isinstance(list1, list) or not isinstance(list2, list):
        return np.nan  # Invalid comparison
        
    # The maximum token set ratio for each item in list 1 when compare to each item in list 2
    scores = [
        max(fuzz.token_set_ratio(str(item1).lower(), str(item2).lower()) for item2 in list2) 
        for item1 in list1
    ]

    # Aggregate methods:
    if method == "average":
        return np.mean(scores) if scores else np.nan
    elif method == "max":
        return np.max(scores) if scores else np.nan
    else:
        raise ValueError("Invalid method. Choose 'average' or 'max'")


@ray.remote
def compute_normalized_token_set_ratio(list1, list2, method="average"):
    """
    Compute token set ratio normalized by the longer text length.
    Handles empty lists and ensures normalized scores are capped at 100.

    Parameters:
    ----------
    list1 : list of str
        The first list of strings to compare.
    list2 : list of str
        The second list of strings to compare.
    method : str, optional, default="average"
        Aggregation method for token set ratios:
        - "average": Returns the mean of the maximum token set ratios.
        - "max": Returns the maximum token set ratio.

    Returns:
    -------
    float
        The normalized token set ratio:
        - 100 if both lists are empty.
        - 0 if one list is empty and the other is not.
        - A normalized token set ratio otherwise.

    Notes:
    ------
    - Normalization ensures scores reflect similarity relative to the longer text length.
    - Scores are capped at 100 to avoid inflation.
    """
    # Handle empty list scenarios
    if not list1 or not list2:
        return 0 if list1 != list2 else 100

    # Compute the token length of the combined text for each list
    length1 = len(" ".join(list1).split())
    length2 = len(" ".join(list2).split())
    max_length = max(length1, length2, 1)  # avoid zero division

    # Compute raw token set ratios
    scores = [
        max(fuzz.token_set_ratio(item1.lower(), item2.lower()) for item2 in list2)
        for item1 in list1
    ]

    # Aggregate raw scores
    raw_score = np.mean(scores) if method == "average" else np.max(scores)

    # Normalize the raw score
    normalized_score = (raw_score / max_length) * 100

    return min(normalized_score, 100)  # Cap the score at 100 to avoid inflation

@ray.remote
def compute_normalized_levenshtein(list1, list2, method="average"):
    """
    Compute Levenshtein distance normalized by the longer text length.
    Handles empty lists and ensures similarity scores are between 0 and 1.

    Parameters:
    ----------
    list1 : list of str
        The first list of strings to compare.
    list2 : list of str
        The second list of strings to compare.
    method : str, optional, default="average"
        Aggregation method:
        - "average": Returns the mean of the similarity scores.
        - "max": Returns the maximum similarity score.

    Returns:
    -------
    float
        Normalized similarity score between 0 and 1:
        - 1.0 if both lists are identical or empty.
        - 0.0 if the lists are completely different.
    """
    if not list1 or not list2:
        return 0 if list1 != list2 else 1.0

    # Calculate Levenshtein distances normalized by max string length
    scores = [
        max(1 - (distance(item1, item2) / max(len(item1), len(item2), 1)) for item2 in list2)
        for item1 in list1
    ]

    # Aggregate scores
    if method == "average":
        return np.mean(scores) if scores else 0.0
    elif method == "max":
        return np.max(scores) if scores else 0.0
    else:
        raise ValueError("Invalid method. Choose 'average' or 'max'.")



@ray.remote
def compute_exact_match(list1, list2):
    """
    Compute exact match between two lists of strings.

    Parameters:
    ----------
    list1 : list of str
        The first list of strings to compare.
    list2 : list of str
        The second list of strings to compare.

    Returns:
    -------
    int
        1 if the lists are identical, 0 otherwise.
    """
    if list1 == list2:
        return 1  # Exact match
    return 0  # No match


def calculate_comparison_metrics_ray(df1, df2, method="average"):
    """
    Compare two DataFrames column-wise where each cell contains a list of strings.
    Returns a dictionary of DataFrames with token set ratio, Levenshtein distance, and exact match scores.

    Parameters:
    ----------
    - df1, df2: Input DataFrames.
    - method: Scoring method ('average' or 'max'). Determines how scores are aggregated.

    Returns:
    -------
    dict
        A dictionary containing DataFrames for token set ratio, Levenshtein distance, and exact match scores.
    """
    # Align DataFrames to ensure they have the same shape
    df1_aligned, df2_aligned = df1.align(df2, join="outer", axis=1)
    assert df1_aligned.shape == df2_aligned.shape, "DataFrames do not have the same shape for comparison."

    # Replace NaN with empty lists
    df1_aligned = df1_aligned.map(lambda x: x if isinstance(x, list) else [])
    df2_aligned = df2_aligned.map(lambda x: x if isinstance(x, list) else [])

    # Initialize DataFrames to store results
    token_set_ratio_df = pd.DataFrame(index=df1_aligned.index, columns=df1_aligned.columns)
    levenshtein_df = pd.DataFrame(index=df1_aligned.index, columns=df1_aligned.columns)
    exact_match_df = pd.DataFrame(index=df1_aligned.index, columns=df1_aligned.columns)

    # Create a dictionary to hold Ray futures
    token_set_futures = {}
    levenshtein_futures = {}
    exact_match_futures = {}

    for col in df1_aligned.columns:
        for idx in df1_aligned.index:
            # Add futures for token set ratio, Levenshtein distance, and exact match
            token_set_futures[(idx, col)] = compute_normalized_token_set_ratio.remote(
                df1_aligned.at[idx, col], df2_aligned.at[idx, col], method
            )
            levenshtein_futures[(idx, col)] = compute_normalized_levenshtein.remote(
                df1_aligned.at[idx, col], df2_aligned.at[idx, col], method
            )
            exact_match_futures[(idx, col)] = compute_exact_match.remote(
                df1_aligned.at[idx, col], df2_aligned.at[idx, col]
            )

    # Gather results from Ray futures
    for (idx, col), future in token_set_futures.items():
        token_set_ratio_df.at[idx, col] = ray.get(future)
    for (idx, col), future in levenshtein_futures.items():
        levenshtein_df.at[idx, col] = ray.get(future)
    for (idx, col), future in exact_match_futures.items():
        exact_match_df.at[idx, col] = ray.get(future)

    token_set_ratio_df["case_average"] = token_set_ratio_df.mean(axis=1)
    levenshtein_df["case_average"] = levenshtein_df.mean(axis=1)
    exact_match_df["case_average"] = exact_match_df.mean(axis=1)
    
    # Add overall average as a new row
    token_set_ratio_df.loc["category_average"] = token_set_ratio_df.mean(axis=0)
    levenshtein_df.loc["category_average"] = levenshtein_df.mean(axis=0)
    exact_match_df.loc["category_average"] = exact_match_df.mean(axis=0)
    
    # Return the three DataFrames
    return token_set_ratio_df, levenshtein_df, exact_match_df
