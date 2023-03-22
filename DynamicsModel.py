import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import scatter
from torch_geometric.nn import MetaLayer

class DynamicsModel(nn.Module):
    def __init__(self, device, hidden_dim, node_feature_size, global_dim, action_dim):
        super(DynamicsModel, self).__init__()
        self.device = device
        self.node_feature_size = node_feature_size
        self.global_dim = global_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.dynamics_mlp = nn.Sequential(nn.Linear(self.node_feature_size+self.action_dim, self.hidden_dim), 
                              nn.BatchNorm1d(self.hidden_dim),
                              nn.ReLU(), 
                              nn.Linear(self.hidden_dim, self.global_dim))
        
    def forward(self, input_data, action, target_node):
        x = input_data['x'].to(self.device)
        #edge_index = input_data['edge_index'].to(self.device)
        #edge_attr = input_data['edge_attr'].to(self.device)
        #u = input_data['u'].to(self.device)
        batch = input_data['batch'].to(self.device)
        batch_list = []
        for i in range(batch[-1]+1):
            batch_list.append(x[(batch==i).nonzero(as_tuple=False).reshape(-1),:])
        x = torch.stack(batch_list).to(self.device)
        #target_node = (target_node==1).nonzero(as_tuple=False).reshape(-1).to(self.device) #target node에서 1인 부분의 index값
        #target_node = (target_node==1).nonzero(as_tuple=False)[:,-1].to(self.device)
        #x_target_node = x[:, target_node, :]
        #x_target_node = scatter(x, target_node, dim=1, reduce='mean')
        target_node = target_node.unsqueeze(-1).permute(0,2,1).to(self.device)
        x_target_node = torch.matmul(target_node, x).squeeze(1).to(self.device)

        action = action.to(self.device)
        dynamics_emb = torch.cat([action, x_target_node], dim=1)
        return self.dynamics_mlp(dynamics_emb)
    

class EdgeBlock(torch.nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(EdgeBlock, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim


        self.edge_mlp = nn.Sequential(nn.Linear(2*self.node_feature_size + self.edge_feature_size + self.global_dim, self.hidden_dim), 
                                      #nn.Linear(2*self.node_feature_size + self.edge_feature_size, self.hidden_dim), 
                                      #BatchNorm1d(hidden),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_dim, self.edge_feature_size))

    def forward(self, src, dest, edge_attr, u, batch):
        #out = torch.cat([src, dest, edge_attr], 1)
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
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
        
        self.node_mlp_2 = nn.Sequential(nn.Linear(self.node_feature_size+self.hidden_dim+self.global_dim, self.hidden_dim),
                                        #nn.Linear(self.node_feature_size+self.hidden_dim, self.hidden_dim),
                                        nn.BatchNorm1d(self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.node_feature_size))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter(out, col, dim=0, dim_size=x.size(0), reduce='mean')
        out = torch.cat([x, out, u[batch]], dim=1)
        #out = torch.cat([x, out], dim=1)
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
        self.global_mlp = nn.Sequential(nn.Linear(self.node_feature_size+self.global_dim, self.hidden_dim),
                                        nn.BatchNorm1d(self.hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_dim, self.global_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        #out = scatter(x, batch, dim=0, reduce='mean')
        out = torch.cat([u, scatter(x, batch, dim=0, reduce='mean')], dim=1)
        #print("EdgeBlock forward shape ", out.shape)  
        # EdgeBlock forward shape  torch.Size([2, 74])
        # mat1 and mat2 shapes cannot be multiplied (2x74 and 64x64)
        return self.global_mlp(out)

#### Dynamcis test
class GNNEncoder(nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(GNNEncoder, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim
        
        self.gnn1 = MetaLayer(EdgeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              NodeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              GlobalBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim))
        
        self.gnn2 = MetaLayer(EdgeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              NodeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              GlobalBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim))
        
        self.gnn3 = MetaLayer(EdgeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              NodeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              GlobalBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim))

    def forward(self, input_data, dynamics_emb):
        x = input_data['x'].to(self.device)
        edge_index = input_data['edge_index'].to(self.device).type(torch.long)
        edge_attr = input_data['edge_attr'].to(self.device)
        #u = input_data['u'].to(self.device)
        batch = input_data['batch'].to(self.device)

        if dynamics_emb == None :
            dynamics_emb = torch.zeros((batch[-1]+1, self.global_dim)).to(self.device)

        x, edge_attr, u = self.gnn1(x=x, edge_index=edge_index, edge_attr=edge_attr, u=dynamics_emb, batch=batch)
        x, edge_attr, u = self.gnn2(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)
        x, edge_attr, u = self.gnn3(x=x, edge_index=edge_index, edge_attr=edge_attr, u=u, batch=batch)
        

        return x, edge_attr, u, batch
    

#### Dynamcis test
class GNNEncoder_test(nn.Module):
    def __init__(self, device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim):
        super(GNNEncoder_test, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.global_dim = global_dim
        
        self.gnn1 = MetaLayer(EdgeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              NodeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              GlobalBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim))
        
        self.gnn2 = MetaLayer(EdgeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              NodeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              GlobalBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim))
        
        self.gnn3 = MetaLayer(EdgeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              NodeBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim),
                              GlobalBlock(self.device, self.hidden_dim, self.num_action, self.node_feature_size, self.edge_feature_size, self.global_dim))

    def forward(self, state_data, dynamics_emb):
        state_x = state_data['x'].to(self.device)
        state_edge_index = state_data['edge_index'].to(self.device).type(torch.long)
        state_edge_attr = state_data['edge_attr'].to(self.device)
        #u = input_data['u'].to(self.device)
        state_batch = state_data['batch'].to(self.device)




        if dynamics_emb == None :
            dynamics_emb = torch.zeros((state_batch[-1]+1, self.global_dim)).to(self.device)

        state_x, state_edge_attr, state_u = self.gnn1(x=state_x, edge_index=state_edge_index, edge_attr=state_edge_attr, u=dynamics_emb, batch=state_batch)
        state_x, state_edge_attr, state_u = self.gnn2(x=state_x, edge_index=state_edge_index, edge_attr=state_edge_attr, u=state_u, batch=state_batch)
        state_x, state_edge_attr, state_u = self.gnn3(x=state_x, edge_index=state_edge_index, edge_attr=state_edge_attr, u=state_u, batch=state_batch)
        

        return state_x, state_edge_attr, state_u, state_batch