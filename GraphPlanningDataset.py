import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
import pickle

## Dataset
class GraphPlanningDataset(Dataset):
    def __init__(self, search_path):
        self.search_path = os.path.join(os.getcwd(),'datasets',search_path)
        self.data_list = os.listdir(self.search_path)

    # Data size return
    def __len__(self): 
        return len(self.data_list)


    # Sampling one specific data from dataset
    def __getitem__(self, index): 
        loaded_data = {}
        with open(os.path.join(self.search_path, self.data_list[index]), "rb") as file:
            loaded_data = pickle.load(file)

        return loaded_data
