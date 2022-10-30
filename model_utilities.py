import csv
from torch.utils.data import Dataset


class MLDataset(Dataset):
    "Assumes csv file. Indices should be a dict of strings returning integers"
    
    def __init__(self, kg_path, entities_index, relations_index, mask=None):
        self.entities_index = entities_index
        self.relations_index = relations_index
        
        with open(kg_path, "r") as f:
            self.kg = [line.strip("\n").split("::") for line in f]
            
        if mask == "feedback":
            with open(kg_path, "r") as f:
                self.kg = []
                for line in f:
                    candidate = line.strip("\n").split("::")
                    if candidate[1] == 'feedback':
                        self.kg.append(candidate)
            
    def __len__(self):
        return len(self.kg)
    
    def __getitem__(self, index):
        h, l, t = self.kg[index]
        try:
            return int(self.entities_index[h]), int(self.relations_index[l]), int(self.entities_index[t])
        except:
            # print(h, l, t)
            print("Exception __getitem__")
            return len(entities_index), len(relations_index), len(entities_index)
        
