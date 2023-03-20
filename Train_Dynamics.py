from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from GraphPlanningDataset import GraphPlanningDataset
import pickle
from DynamicsModel import *
import os

def train_dynamics(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir):
    dynamics_model = DynamicsModel(device, node_feature_size, global_dim, num_action)
    gnn_encoder = GNNEncoder(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim)

    dynamics_model.to(device)
    gnn_encoder.to(device)

    model_name = [data_dir, hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(), "result","dynamics", "_".join(list(map(str, model_name))))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_dataset = GraphPlanningDataset(os.path.join(data_dir,'train'))
    val_dataset = GraphPlanningDataset(os.path.join(data_dir,'val'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam([{'params': dynamics_model.parameters()},
                                  {'params': gnn_encoder.parameters()}], lr = lr)
    
    loss_cos_sim = nn.CosineEmbeddingLoss().to(device)

    for param in dynamics_model.parameters():
        param.requires_grad = True

    best_loss = 10000
    
    loss_data = {"epoch":[],
                 "loss":{"total":{"train":[],
                                  "val":[]}}}

    #train
    for epoch in range(num_epoch):
        print("#############################")
        print("epoch number {}".format(epoch+1))
        dynamics_model.train()
        gnn_encoder.train()

        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(train_loader):
            state, goal, target = data

            dynamics_emb = dynamics_model(state, target['action'], target['object'])
            
            state_x, state_edge_attr, state_u, state_batch = gnn_encoder(state, dynamics_emb)
            goal_x, goal_edge_attr, goal_u, goal_batch = gnn_encoder(goal, None)

            
            sim_label = torch.ones(state_u.size(0), 1)
            L_dynamics = loss_cos_sim(state_u, goal_u, sim_label)
            L_dynamics.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += L_dynamics.item()
            last_loss = running_loss/(i+1)

        dynamics_model.eval()
        gnn_encoder.eval()

        val_running_loss = 0.0
        val_last_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_state, val_goal, val_target = data

                val_dynamics_emb = dynamics_model(val_state, val_target['action'], val_target['object'])
                
                state_x, state_edge_attr, state_u, state_batch = gnn_encoder(val_state, val_dynamics_emb)
                goal_x, goal_edge_attr, goal_u, goal_batch = gnn_encoder(val_goal, None)

                
                sim_label = torch.ones(state_u.size(0), 1)
                L_dynamics = loss_cos_sim(state_u, goal_u, sim_label)

                val_running_loss += L_dynamics.item()
                val_last_loss = val_running_loss/(i+1)


        print("\nTotal Loss\ttrain:{:01.4f}\tval:{:01.4f}".format(last_loss, val_last_loss))

        loss_data['epoch'].append(epoch)

        loss_data['loss']['total']['train'].append(last_loss)
        loss_data['loss']['total']['val'].append(val_last_loss)

        if val_last_loss < best_loss:
            best_loss = val_last_loss
            torch.save(dynamics_model.state_dict(), model_path + '/DynamicsModel_{}.pt'.format(epoch))
            torch.save(dynamics_model.state_dict(), model_path + '/DynamicsModel_best.pt')
            torch.save(gnn_encoder.state_dict(), model_path + '/GNNEncoder_{}.pt'.format(epoch))
            torch.save(gnn_encoder.state_dict(), model_path + '/GNNEncoder_best.pt')


        #save loss record
        file_path = os.path.join(model_path, 'dynamics_loss_data')
        with open(file_path, "wb") as outfile:
            pickle.dump(loss_data, outfile)