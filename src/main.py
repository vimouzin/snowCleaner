
__version__ = "1.0.0"

import itertools

import pandas as pd
from snowcleaner.model.pair_generator import calculate_similarity_scores_for_pairs, block_records_with_threshold
from snowcleaner.model.train_model import train_model, predict

features = ["first_name", "last_name", "address", "city", "state", "country"]

df = pd.read_csv('exampledatasets/data.csv', dtype=str)

# Accumulate similarity scores
similarity_scores = []

# Iterate over unique pairs of rows
for (index, row), (index_1, row_2) in itertools.combinations(df.iterrows(), 2):
    # Convert Pandas Series to dictionaries
    row_1_dict = row.to_dict()
    row_2_dict = row_2.to_dict()
    scores = calculate_similarity_scores_for_pairs(features, row_1_dict, row_2_dict)
    similarity_scores.append(scores)

# Create DataFrame from similarity_scores
result_df = pd.read_csv('exampledatasets/test_data_final.csv')


results = block_records_with_threshold(
    result_df,
    {
        "first_name": 0.6,
        "last_name": 0.6,
        "address": 0.6,
        "city": 0.0,
        "state": 0.0,
        "country": 0.0,
        "pair": 0.0
    }
)
print(results)
model = train_model(results)
predict(model,result_df)

#result_df.to_csv("test_6763.csv")