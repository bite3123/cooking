import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
import pickle

## Dataset
class StackingDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path#/stacking_dataset

        # Search path
        search_path = os.path.join(os.getcwd(), self.root_path)
        self.search_path = search_path
        #print("search_path:",search_path)

    # Data size return
    def __len__(self): 
        #return 960
        return len(os.listdir(self.search_path))


    # Sampling one specific data from dataset
    def __getitem__(self, index): 
        loaded_data = {}
        with open(self.search_path + "/stacking_" + str(index), "rb") as file:
            loaded_data = pickle.load(file)
        #print(loaded_data['info'])
        '''
        x = loaded_data['input']['state']['x']
        edge_index = loaded_data['input']['state']['edge_index']
        edge_attr = loaded_data['input']['state']['edge_attr']
        '''
        x = loaded_data['input']['state']['x']
        edge_index = loaded_data['input']['key']['edge_index']
        edge_attr = loaded_data['input']['key']['edge_attr']
        
        key_node = loaded_data['input']['key']['key_node']
        
        input_data = Data(x, edge_index, edge_attr)   
            
        #input_data = loaded_data['input']
        target_data = loaded_data['target']
       

        


        return input_data, target_data, key_node


#total_dataset = StackingDataset('stacking_dataset')
#train_dataset, test_dataset = random_split(total_dataset, [0.8, 0.2])
