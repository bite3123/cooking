from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, GINEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer

class EdgeBlock(torch.nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(EdgeBlock, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim


        self.edge_mlp = nn.Sequential(nn.Linear(2*self.node_feature_size + self.edge_feature_size, self.hidden_dim), 
                            #BatchNorm1d(hidden),
                            nn.ReLU(),
                            nn.Linear(self.hidden_dim, self.edge_feature_size))

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        #out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)
    
class NodeBlock(torch.nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(NodeBlock, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim

        self.node_mlp_1 = nn.Sequential(nn.Linear(self.node_feature_size+self.edge_feature_size, self.hidden_dim), 
                              nn.BatchNorm1d(self.hidden_dim),
                              nn.ReLU(), 
                              nn.Linear(self.hidden_dim, self.hidden_dim))
        
        self.node_mlp_2 = nn.Sequential(nn.Linear(self.node_feature_size+self.hidden_dim, self.hidden_dim), 
                              nn.BatchNorm1d(self.hidden_dim),
                              nn.ReLU(), 
                              nn.Linear(self.hidden_dim, self.node_feature_size))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')
        out = torch.cat([x, out], dim=1)
        #out = torch.cat([x, out, u[batch]], dim=1)
        #print("EdgeBlock forward shape ", out.shape)   
        # EdgeBlock forward shape  torch.Size([72, 78])
        # mat1 and mat2 shapes cannot be multiplied (72x78 and 68x64)
        return self.node_mlp_2(out)

class GlobalBlock(torch.nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(GlobalBlock, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim

        #self.global_mlp = nn.Sequential(nn.Linear(hidden, hidden),                               
        self.global_mlp = nn.Sequential(nn.Linear(self.node_feature_size, self.hidden_dim),                               
                              nn.BatchNorm1d(self.hidden_dim),
                              nn.ReLU(), 
                              nn.Linear(self.hidden_dim, self.global_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter(x, batch, dim=0, reduce='mean')
        #out = torch.cat([
        #    u,
        #    scatter(x, batch, dim=0, reduce='mean'),
        #], dim=1)
        #print("EdgeBlock forward shape ", out.shape)  
        # EdgeBlock forward shape  torch.Size([2, 74])
        # mat1 and mat2 shapes cannot be multiplied (2x74 and 64x64)
        return self.global_mlp(out)

#### ActionModel
class ActionModel(nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(ActionModel, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim
        
        self.gnn1 = MetaLayer(EdgeBlock(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim),
                              NodeBlock(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim),
                              GlobalBlock(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim))
        
        self.gnn2 = MetaLayer(EdgeBlock(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim),
                              NodeBlock(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim),
                              GlobalBlock(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim))
        
        self.gnn3 = MetaLayer(EdgeBlock(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim),
                              NodeBlock(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim),
                              GlobalBlock(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim))  
         
        self.action_layers = nn.Sequential(
            nn.Linear(self.node_feature_size, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_action),
            #nn.Sigmoid()
            #nn.BatchNorm1d(self.num_action),
            #nn.ReLU(),
            #nn.Sigmoid(),
        )
        self.x_embedding= nn.Sequential(
            nn.Linear(self.node_feature_size, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.node_feature_size),
            nn.ReLU()
            #nn.BatchNorm1d(self.num_action),
            #nn.ReLU(),
            #nn.Sigmoid(),
        )
        self.node_layers = nn.Sequential(
            nn.Linear(self.node_feature_size, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 33),
            #nn.Sigmoid()
            #nn.BatchNorm1d(self.num_action),
            #nn.ReLU(),
            #nn.Sigmoid(),
        )

    def forward(self, input_data):
        x = input_data['x'].to(self.device)
        edge_index = input_data['edge_index'].to(self.device).type(torch.long)
        edge_attr = input_data['edge_attr'].to(self.device)
        #u = input_data['u'].to(self.device)
        batch = input_data['batch'].to(self.device)

        x, edge_attr, u = self.gnn1(x=x, edge_index=edge_index, edge_attr=edge_attr, u=None, batch=batch)
        x, edge_attr, u = self.gnn2(x=x, edge_index=edge_index, edge_attr=edge_attr, u=None, batch=batch)
        x, edge_attr, u = self.gnn3(x=x, edge_index=edge_index, edge_attr=edge_attr, u=None, batch=batch)

        action_emb_list = []
        node_score_list = []
        batch_list = []
        for i in range(input_data['batch'][-1]+1):
            x_per_batch = x[(input_data['batch']==i).nonzero(as_tuple=False).reshape(-1),:]

            batch_list.append(x[(input_data['batch']==i).nonzero(as_tuple=False).reshape(-1),:])
        x = torch.stack(batch_list).to(self.device)

        x_emb = x.mean(axis=1)#아무연결없는노드 빼는법 있을까?
        action_prob = self.action_layers(x_emb)
        node_score = self.node_layers(x_emb)

        return action_prob, node_score
