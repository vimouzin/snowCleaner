
__version__ = "1.0.0"

import itertools

import pandas as pd

from snowcleaner.model.pair_generator import calculate_similarity_scores_for_pairs, block_records_with_threshold
from snowcleaner.model.train_model import train_model, predict
from snowcleaner.clusteridgenerator.clusteridgenerator import generate_cluster_ids

features = ["first_name", "last_name", "address", "city", "state", "country"]

df = pd.read_csv('exampledatasets/data.csv')

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
result_df = pd.read_csv('exampledatasets/training_data.csv')


results = block_records_with_threshold(
    result_df,
    {
        "first_name": 0.2,
        "last_name": 0.1,
        "address": 0.1,
        "city": 0.1,
        "state": 0.1,
        "country": 0.1
    }
)

predict_record = pd.read_csv('exampledatasets/training_data.csv')
predict_record = predict_record.drop(columns=['pair'])

training_data = results.drop(columns=['pk','pk_2'])

model = train_model(training_data)
final_results = predict(model,result_df.drop(columns="pair"))
df1 = generate_cluster_ids(final_results)
df1 = df1[df1['group_id'] != '']


merged_df_pk = pd.merge(df, df1, left_on='pk', right_on='pk_2', how='left')

# Merge based on 'pk_2'
merged_df_pk_2 = pd.merge(df, df1, left_on='pk', right_on='pk', how='left')

# Combine the results
merged_df = merged_df_pk.combine_first(merged_df_pk_2)

# Drop redundant columns
merged_df = merged_df[features + ["group_id","pk"]]
merged_df.to_csv("results.csv")

