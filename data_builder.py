"""Build kg graph and create indices.

Reads in a list of subgraphs in terms of edgelists, 
combines them in (entity, relation, entity) tuples and writes to file.
Additionally, creates indices for the entities and relations.
"""

import os
import csv

dataset_root = "data/"
subgraphs_dir =  dataset_root + "subgraphs/"
kg_dir = dataset_root + "/kg/"

entities, relations = [], []

# For get (entity, relation, entity) from each subgraph file.
with open(kg_dir + "kg.csv", "w") as kg_file:
    for f in os.listdir(subgraphs_dir):
        relation_name = f.replace(".edgelist", "")
        with open(f"{subgraphs_dir}/{f}", "r") as edgelists:
            for edge in edgelists:
                edge_components = edge.strip("\n").split(" ")
                kg_file.write(f"{edge_components[0]}::{relation_name}::"\
                              f"{edge_components[1]}\n")
                
                # Append entities and relations.
                entities.append(edge_components[0])
                relations.append(relation_name)
                entities.append(edge_components[1])
                
                
# Indices for entities and relations.
with open(kg_dir + "entities_index.csv", "w") as f:
    for idx, entity in enumerate(list(set(entities))):
        f.write(f"{entity}\t{idx}\n")
        
with open(kg_dir + "relations_index.csv", "w") as f:
    for idx, relation in enumerate(list(set(relations))):
        f.write(f"{relation}\t{idx}\n")



def get_entities_and_relations():
    return entities, relations


def create_index_dictionary(file_path):
    "Creates dictionary as index from file"
    with open(file_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        index = {rows[0]:int(rows[1]) for rows in reader}
        
    return index
