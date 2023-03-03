from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, GINEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
#### ActionModel

class ActionModel(nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size):
        super(ActionModel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size

        self.conv1 = GINEConv(nn=nn.Sequential(nn.Linear(self.node_feature_size, self.hidden_dim),
                                                nn.BatchNorm1d(self.hidden_dim),
                                                #nn.ReLU(),
                                                #nn.Linear(self.hidden_dim, self.hidden_dim),
                                                #nn.BatchNorm1d(self.hidden_dim),
                                                nn.ReLU(),),
                               edge_dim=self.edge_feature_size)
        self.conv2 = GINEConv(nn=nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.BatchNorm1d(self.hidden_dim),
                                                #nn.LeakyReLU(),
                                                #nn.Linear(self.hidden_dim, self.hidden_dim),
                                                #nn.BatchNorm1d(self.hidden_dim),
                                                nn.ReLU(),
                                                nn.Sigmoid()),
                               edge_dim=self.edge_feature_size)
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
        
        x = input_data['x'].to(self.device)
        edge_index = input_data['edge_index'].to(self.device)
        edge_attr = input_data['edge_attr'].to(self.device)

        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        
        batch_list = []
        for i in range(input_data['batch'][-1]+1):
            batch_list.append(x[(input_data['batch']==i).nonzero(as_tuple=False).reshape(-1),:])
        x = torch.stack(batch_list).to(self.device)

        action_input_emb = x.mean(axis=1)
        #softmax = nn.Softmax(dim=1).to(device)
        #action_prob = softmax(self.action_layers(action_input_emb))
        action_prob = self.action_layers(action_input_emb)

        return action_prob
    

#### ActionModel

class ActionModel2(nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size):
        super(ActionModel2, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        
        self.conv1 = GINEConv(nn=nn.Sequential(nn.Linear(self.node_feature_size, self.hidden_dim),
                                                nn.BatchNorm1d(self.hidden_dim),
                                                #nn.ReLU(),
                                                #nn.Linear(self.hidden_dim, self.hidden_dim),
                                                #nn.BatchNorm1d(self.hidden_dim),
                                                nn.ReLU(),),
                               edge_dim=self.edge_feature_size)
        self.conv2 = GINEConv(nn=nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.BatchNorm1d(self.hidden_dim),
                                                #nn.LeakyReLU(),
                                                #nn.Linear(self.hidden_dim, self.hidden_dim),
                                                #nn.BatchNorm1d(self.hidden_dim),
                                                nn.ReLU(),
                                                nn.Sigmoid()),
                               edge_dim=self.edge_feature_size)
        
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

        self.node_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        self.sig = nn.Sigmoid()

    def forward(self, input_data):
        x = input_data['x'].to(self.device)
        edge_index = input_data['edge_index'].to(self.device)
        edge_attr = input_data['edge_attr'].to(self.device)

        x = self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        action_emb_list = []
        node_score_list = []
        batch_list = []
        for i in range(input_data['batch'][-1]+1):
            x_per_batch = x[(input_data['batch']==i).nonzero(as_tuple=False).reshape(-1),:]
            node_score_per_batch = []
            
            for node_emb in x_per_batch:
                node_score_per_batch.append(self.sig(self.node_layers(node_emb)))
            node_score_per_batch = torch.cat(node_score_per_batch)

            action_input_emb_per_batch = x_per_batch.mean(axis=0)

            node_score_list.append(node_score_per_batch)
            action_emb_list.append(action_input_emb_per_batch)
            batch_list.append(x[(input_data['batch']==i).nonzero(as_tuple=False).reshape(-1),:])
        x = torch.stack(batch_list).to(self.device)
        action_input_emb = torch.stack(action_emb_list).to(self.device)
        node_score = torch.stack(node_score_list).to(self.device)

        #softmax = nn.Softmax(dim=1).to(device)
        #action_prob = softmax(self.action_layers(action_input_emb))
        action_prob = self.action_layers(action_input_emb)


        return action_prob, node_score
  
