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

# Basically the same as the baseline except we pass edge features 

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"


#### Action input embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#### ActionModel

class ActionModel(nn.Module):
    def __init__(self, hidden_dim, num_action, node_feature_size, edge_feature_size):
        super(ActionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size

        self.convs = [GINEConv(nn=nn.Sequential(nn.Linear(self.node_feature_size, self.hidden_dim),
                                                nn.BatchNorm1d(self.hidden_dim),
                                                #nn.ReLU(),
                                                #nn.Linear(self.hidden_dim, self.hidden_dim),
                                                #nn.BatchNorm1d(self.hidden_dim),
                                                nn.ReLU(),),
                               edge_dim=self.edge_feature_size),
                      GINEConv(nn=nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.BatchNorm1d(self.hidden_dim),
                                                #nn.LeakyReLU(),
                                                #nn.Linear(self.hidden_dim, self.hidden_dim),
                                                #nn.BatchNorm1d(self.hidden_dim),
                                                nn.ReLU(),
                                                nn.Sigmoid()),
                               edge_dim=self.edge_feature_size)]
        
        self.action_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_action),
            nn.Sigmoid()
            #nn.BatchNorm1d(self.num_action),
            #nn.ReLU(),
            #nn.Sigmoid(),
        )

    def forward(self, input_data):
        
        x = input_data['x']
        edge_index = input_data['edge_index']
        edge_attr = input_data['edge_attr']

        x.to(device)
        edge_index.to(device)
        edge_attr.to(device)

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            #x = F.dropout(x, training = self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        
        batch_list = []
        for i in range(input_data['batch'][-1]+1):
            batch_list.append(x[(input_data['batch']==i).nonzero(as_tuple=False).reshape(-1),:])
        x = torch.stack(batch_list)

        action_input_emb = x.mean(axis=1)
        #softmax = nn.Softmax(dim=1).to(device)
        #action_prob = softmax(self.action_layers(action_input_emb))
        action_prob = self.action_layers(action_input_emb)

        return action_prob