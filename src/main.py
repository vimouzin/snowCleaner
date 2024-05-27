
__version__ = "1.0.0"

import itertools

import pandas as pd

from snowcleaner.model.pair_generator import calculate_similarity_scores_for_pairs, block_records_with_threshold
from snowcleaner.model.train_model import train_model, predict
from snowcleaner.clusteridgenerator.clusteridgenerator import generate_cluster_ids
import streamlit as st

st.set_page_config(page_title="SnowCleaner",layout="wide")
st.subheader("Welcome to SnowCleaner")

features = ["first_name", "last_name", "address", "city", "state", "country"]

df = pd.read_csv('exampledatasets/data.csv')
st.dataframe(df, use_container_width=True)

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


pk1 = df1[["pk","group_id"]]
pk2 = df1[["pk_2","group_id"]]

pk1 = pk1.drop_duplicates(subset=['pk'])
pk2 = pk2.drop_duplicates(subset=['pk_2'])

merged_df_pk_2 = pd.merge(df, pk1, left_on='pk', right_on='pk', how='left')

merged_df_pk = pd.merge(df, pk2, left_on='pk', right_on='pk_2', how='left')


merged_df = merged_df_pk.combine_first(merged_df_pk_2)


# Drop redundant columns
merged_df = merged_df[["pk","group_id"] + features]
merged_df.to_csv("results.csv")

st.dataframe(merged_df, use_container_width=True)

