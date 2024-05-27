# clusteridgenerator/clusteridgenerator.py

import networkx as nx

def generate_cluster_ids(result_df):
    # Create an undirected graph
    G = nx.Graph()

    # Add edges to the graph for each row where predictions are positive
    for index, row in result_df[result_df['predictions'] > 0].iterrows():
        G.add_edge(row['pk'], row['pk_2'])

    # Find all connected components (subgraphs)
    connected_components = list(nx.connected_components(G))

    # Create a dictionary to hold group IDs
    group_dict = {}
    for group_id, component in enumerate(connected_components):
        for node in component:
            group_dict[node] = group_id

    # Create a new column in the DataFrame for group ID
    result_df['group_id'] = result_df.apply(lambda x: group_dict.get(x['pk'], -1) if x['predictions'] > 0 else '', axis=1)

    return result_df