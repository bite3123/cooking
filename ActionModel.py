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
    def __init__(self, hidden_dim, num_action, node_feature_size, edge_feature_size):
        super(ActionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        
        self.convs = [GATConv(in_channels=self.node_feature_size, out_channels=self.hidden_dim, edge_dim=self.edge_feature_size),
                      GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, edge_dim=self.edge_feature_size)]
        
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

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Data format must match! 
        # Type 1) x : float32, edge_index : int64, edge_attr: float32  
        # print(type(x),type(edge_index),type(edge_attr))

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training = self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        
        # print("x",x)
        action_input_emb = x.mean(axis=0)      # x feature를 합치는 과정 / 현재는 mean으로 (추후 변경 예정)
        # print("actopm=input",action_input_emb)
        self.node_features = x
                
        softmax = nn.Softmax(dim=0)
        action_prob = softmax(self.action_layers(action_input_emb))
       
        # action_prob = self.action_layers(action_input_emb)
    
    
        # action_prob = nn.Softmax(self.action_layers(action_input_emb))
        
        
        each_node = []
        for feature in self.node_features:  
            # print(feature) # feature size : [8,8]

            sig = nn.Sigmoid()
            # node_scores.append(nn.Sigmoid(self.node_layers(feature)))
            each_node.append(sig(self.node_layers(feature))) #tensor는 append로 합친 후 concat을 해야 list형식의 tensor형태로 가지고 있는다.
        
            node_scores = torch.cat(each_node, dim=0)
        # print("\n[Each node]",each_node)

        return action_prob, node_scores   
  
