from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import  random_split
import torch.optim as optim
from StackingDataset import StackingDataset
from ActionModel import ActionModel
import matplotlib.pyplot as plt
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"

hidden_dim = 256
num_action = 3 # [pick, place, pour]
node_feature_size = 6 #노드 feature 크기
edge_feature_size = 14 # 노드 사이의 relation 종류 개수 [on_right,on_left, in_right, in_left, attach, in-grasp]
batch_size = 1

