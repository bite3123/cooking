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
    def __init__(self, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size):
        super(ActionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.batch_size = batch_size

        self.convs = [GINEConv(nn=nn.Sequential(nn.Linear(self.node_feature_size, self.hidden_dim),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.LeakyReLU(),),
                               edge_dim=self.edge_feature_size),
                      GINEConv(nn=nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.LeakyReLU(),),
                               edge_dim=self.edge_feature_size)]
        
        self.action_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.num_action),
            nn.LeakyReLU(),
        )
        
        self.node_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, input_data, target_data):
        
        x = input_data['x']
        edge_index = input_data['edge_index']
        edge_attr = input_data['edge_attr']



        x.to(device)
        edge_index.to(device)
        edge_attr.to(device)
        #print(input_data)
        #x, edge_index, edge_attr, _, _= input_data
        #print(x)
        #print(edge_index.shape)
        #print(edge_attr.shape)
        #input()

        # Data format must match! 
        # Type 1) x : float32, edge_index : int64, edge_attr: float32  
        # print(type(x),type(edge_index),type(edge_attr))

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training = self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        

        #144X64 -> (9X16)X64 = (num_node X batch)Xhidden_dim
        x = x.reshape(self.batch_size, -1, self.hidden_dim) #batch X node X hidden
        #print(x.shape)


        ############

        target_object = target_data['object']
        object_idx_tensor = (target_object==1).nonzero()[:,1]
        #print(target_object)
        #print(object_idx_tensor)
        #input()
        #for i in len(target_object):
            #if 

        x_selected = []
        for i in range(self.batch_size):
            #print(i.shape)
            #x_selected.append(i[])
            #print(object_idx_tensor[i].item())
            #print(x[i,:,:].shape)
            x_selected.append(x[i,object_idx_tensor[i],:])


        action_input_emb = torch.stack(x_selected, dim=0)
        #print(action_input_emb.shape)
        #input()
        ################

        #action_input_emb = x.mean(axis=1)      # x feature를 합치는 과정 / 현재는 mean으로 (추후 변경 예정)
        #print("actopm=input",action_input_emb)
        #print(action_input_emb.shape) # batch X hidden
        #input()
        softmax = nn.Softmax(dim=1).to(device)
        action_prob = softmax(self.action_layers(action_input_emb))
        #print("action_prob:", action_prob)
        #print(action_prob.shape)
        # action_prob = self.action_layers(action_input_emb)
    
    
        # action_prob = nn.Softmax(self.action_layers(action_input_emb))
        return action_prob
        '''
        
        each_node = []
        x = x.reshape(-1, self.batch_size, self.hidden_dim)
        for feature in x:
            #print(feature.shape) # feature size : batch X hidden

            sig = nn.Sigmoid().to(device)
            # node_scores.append(nn.Sigmoid(self.node_layers(feature)))
            each_node.append(sig(self.node_layers(feature))) #tensor는 append로 합친 후 concat을 해야 list형식의 tensor형태로 가지고 있는다.

        node_scores = torch.cat(each_node, dim=1)
        #print("node_scores", node_scores.shape)
        # print("\n[Each node]",each_node)
        

        return action_prob, node_scores
        '''