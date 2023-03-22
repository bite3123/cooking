import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
import pickle

## Dataset
class ActionDataset(Dataset):
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
        
        x = loaded_data['input']['x']
        edge_index = loaded_data['input']['edge_index']
        edge_attr = loaded_data['input']['edge_attr']

        input_data = Data(x, edge_index, edge_attr)   
        target_data = loaded_data['target']

        info_data = loaded_data['info']

        return input_data, target_data, info_data
    
class DynamicsDataset(Dataset):
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
        
        state_x = loaded_data['state']['x']
        state_edge_index = loaded_data['state']['edge_index']
        state_edge_attr = loaded_data['state']['edge_attr']
        
        goal_x = loaded_data['goal']['x']
        goal_edge_index = loaded_data['goal']['edge_index']
        goal_edge_attr = loaded_data['goal']['edge_attr']        

        state_data = Data(state_x, state_edge_index, state_edge_attr)
        goal_data = Data(goal_x, goal_edge_index, goal_edge_attr)

        target_data = loaded_data['target']

        info_data = loaded_data['info']

        return state_data, goal_data, target_data, info_data
