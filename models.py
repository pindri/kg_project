"""Defines TransE and TransM models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import linalg as LA


class TransE(nn.Module):
    
    def __init__(self, num_entities, num_relations, p, k, gamma):
        super(TransE, self).__init__()
        self.criterion = nn.MarginRankingLoss(margin=gamma, reduction='none')
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.p = p
        self.k = k
        self.gamma = gamma
        self.entities_embedding, self.relations_embedding = self._initialise_embeddings()
        
        
    def _initialise_embeddings(self):
        unif_range = 6/np.sqrt(self.k)
        
        relations_embedding = nn.Embedding(self.num_relations, self.k)
        relations_embedding.weight.data.uniform_(-unif_range, unif_range)
        relations_embedding.weight.data = F.normalize(relations_embedding.weight.data, p=2, dim=1)
        
        entities_embedding = nn.Embedding(self.num_entities, self.k)
        entities_embedding.weight.data.uniform_(-unif_range, unif_range)
        
        return entities_embedding, relations_embedding
    
    def forward(self, training_triplets, corrupted_triplets):
        training_distances = self._transe_interaction(training_triplets)
        corrupted_distances = self._transe_interaction(corrupted_triplets)
        return self.loss(training_distances, corrupted_distances), training_distances, corrupted_distances
    
    def _transe_interaction(self, triplets):
        h = triplets[:, 0]
        l = triplets[:, 1]
        t = triplets[:, 2]
        return (self.entities_embedding(h) + self.relations_embedding(l) -
                self.entities_embedding(t)).norm(p=self.p, dim=1)
        
    def predict(self, triplets):
        return self._transe_interaction(triplets)
        
    def loss(self, training_distances, corrupted_distances):
        target = torch.tensor([-1])
        return self.criterion(training_distances, corrupted_distances, target)


    
class TransM(nn.Module):
    
    def __init__(self, num_entities, num_relations, p, k, gamma):
        super(TransM, self).__init__()
        self.criterion = nn.MarginRankingLoss(margin=gamma, reduction='none')
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.p = p
        self.k = k
        self.gamma = gamma
        self.entities_embedding, self.relations_embedding = self._initialise_embeddings()
        
        
    def _initialise_embeddings(self):
        unif_range = 6/np.sqrt(self.k)
        
        relations_embedding = nn.Embedding(self.num_relations, self.k)
        relations_embedding.weight.data.uniform_(-unif_range, unif_range)
        relations_embedding.weight.data = F.normalize(relations_embedding.weight.data, p=2, dim=1)
        
        entities_embedding = nn.Embedding(self.num_entities, self.k)
        entities_embedding.weight.data.uniform_(-unif_range, unif_range)
        
        return entities_embedding, relations_embedding
    
    def forward(self, training_triplets, corrupted_triplets, relations_weights=None):
        training_distances = self._transm_interaction(training_triplets, relations_weights)
        corrupted_distances = self._transm_interaction(corrupted_triplets, relations_weights)
        return self.loss(training_distances, corrupted_distances), training_distances, corrupted_distances
    
    
    def _transm_interaction(self, triplets, rel_weights):
        h = triplets[:, 0]
        l = triplets[:, 1]
        t = triplets[:, 2]
        norm = (self.entities_embedding(h) + self.relations_embedding(l) - self.entities_embedding(t)).norm(p=self.p, dim=1)
        
        # To assign unitary weights to all triplets for prediction.
        if rel_weights is None:
            rel_weights = torch.ones(norm.size())
        
        interaction = rel_weights * (norm*norm)
        return norm
    
    def predict(self, triplets, rel_weights=None):
        return self._transm_interaction(triplets, rel_weights)
        
    def loss(self, training_distances, corrupted_distances):
        target = torch.tensor([-1])
        return self.criterion(training_distances, corrupted_distances, target)
