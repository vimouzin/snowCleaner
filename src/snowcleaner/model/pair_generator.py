# model/pair_generator.py
import pandas as pd


def jaccard_similarity(str1, str2):
    # Convert the input strings into sets of characters
    set1 = set(str1)
    set2 = set(str2)

    # Calculate the Jaccard similarity coefficient
    intersection = len(set1.intersection(set2))  # Number of common elements
    union = len(set1.union(set2))  # Total number of unique elements

    # Jaccard similarity coefficient formula: |A ∩ B| / |A ∪ B|
    similarity = intersection / union

    # Return the similarity rounded to 4 decimal places
    return round(similarity, 4)


def calculate_similarity_scores_for_pairs(features, pair_1, pair_2):
    """
    Calculate similarity scores for specified features between two records.

    Args:
        features (list): List of features to calculate similarity scores for.
        pair_1 (dict): First record represented as a dictionary.
        pair_2 (dict): Second record represented as a dictionary.

    Returns:
        dict: Dictionary containing feature names as keys and their corresponding similarity scores as values.
    """
    # Initialize an empty dictionary to store similarity scores
    scores = {}

    # Iterate over each feature
    for feature in features:
        # Get the sets of values for the current feature from both records
        set1 = pair_1[feature]
        set2 = pair_2[feature]

        # Calculate the Jaccard similarity score for the current feature and store it in the scores dictionary
        scores[feature] = jaccard_similarity(set1, set2)
        scores["pk"] = pair_1["pk"]
        scores["pk_2"] = pair_2["pk"]
    # Return the dictionary of similarity scores
    return scores



def block_records_with_threshold(data, thresholds):
    """
    Block records based on specified similarity thresholds for each attribute.

    Args:
    - data: Pandas DataFrame containing pairs of records with similarity scores
    - thresholds: Dictionary containing attribute names as keys and corresponding thresholds as values

    Returns:
    - DataFrame containing records that meet the similarity threshold for each attribute
    """
    # Convert columns to appropriate types
    data = data.apply(pd.to_numeric)

    # Make sure columns in the DataFrame match the keys in the thresholds dictionary
    common_columns = set(data.columns) & set(thresholds.keys())
    if len(common_columns) != len(thresholds):
        raise ValueError("Columns in data and thresholds do not match")

    # Apply threshold checks for each attribute
    condition = True
    for column, threshold in thresholds.items():
        condition &= (data[column].apply(float) >= float(threshold))

    return data[condition]
