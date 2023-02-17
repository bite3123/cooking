from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import  random_split
import torch.optim as optim
from StackingDataset import StackingDataset
import pickle
from train import ActionModel
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

hidden_dim = 64
num_action = 3 # [pick, place, pour]
node_feature_size = 6 #노드 feature 크기
edge_feature_size = 13 # 노드 사이의 relation 종류 개수 [on_right,on_left, in_right, in_left, attach, in-grasp]
batch_size = 16

saved_path = "./stacking_model/stacking_model_964"
saved_model = ActionModel()
saved_model.load_state_dict(torch.load())

