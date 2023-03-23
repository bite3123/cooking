from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import  random_split
import torch.optim as optim
from GraphPlanningDataset import *
from DynamicsModel import *
import matplotlib.pyplot as plt
import pickle
import os
def inference_dynamics(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num=None, check_each = False):
    
    model_param = [data_dir, hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(), "result", "dynamics", "_".join(list(map(str, model_param))))

    if infer_num is not None:
        dynamics_model_name = 'DynamicsModel_{}.pt'.format(infer_num)
        gnn_encoder_name = 'GNNEncoder_{}.pt'.format(infer_num)
    else:
        dynamics_model_name = 'DynamicsModel_best.pt'
        gnn_encoder_name = 'GNNEncoder_best.pt'


    with open(os.path.join(model_path, "dynamics_loss_data"), "rb") as file:
        loss_data = pickle.load(file)
    
    epoch_list = loss_data["epoch"]
    
    #Loss Plot
    plt.figure(0)
    plt.figure(figsize=(5, 5))
    plt.suptitle('Loss')

    #Total Loss
    loss_list = []
    val_loss_list = []
    for loss in loss_data['loss']['total']['train']:
        loss_list.append(loss)
    for val_loss in loss_data['loss']['total']['val']:
        val_loss_list.append(val_loss)
    plt.subplot(1,1,1)
    plt.plot(epoch_list, loss_list, label='train')
    plt.plot(epoch_list, val_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('total')

    plt.savefig(os.path.join(model_path, "_".join(list(map(str, model_param))) +'_loss.png'))

    if show_result:
        plt.show()

    dynamics_saved_path = os.path.join(model_path, dynamics_model_name)
    saved_dynamics_model = DynamicsModel(device, hidden_dim, node_feature_size, global_dim, num_action)
    saved_dynamics_model.load_state_dict(torch.load(dynamics_saved_path))
    saved_dynamics_model.to(device)

    gnn_encoder_saved_path = os.path.join(model_path, gnn_encoder_name)
    saved_gnn_encoder = GNNEncoder(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim)
    saved_gnn_encoder.load_state_dict(torch.load(gnn_encoder_saved_path))
    saved_gnn_encoder.to(device)

    for param in saved_dynamics_model.parameters():
        param.requires_grad = False
    for param in saved_gnn_encoder.parameters():
        param.requires_grad = False

    data_test = DynamicsDataset(os.path.join(data_dir,'dynamics','test'))
    data_test_loader = DataLoader(data_test, 1)

    loss_cos_sim = nn.CosineEmbeddingLoss().to(device)

    saved_dynamics_model.eval()
    saved_gnn_encoder.eval()
    for test_data in data_test_loader:
        test_state, test_goal, test_target, test_info = test_data

        test_dynamics_emb = saved_dynamics_model(test_state, test_target['action'], test_target['object'])
        
        state_x, state_edge_attr, state_u, state_batch = saved_gnn_encoder(test_state, test_dynamics_emb)
        goal_x, goal_edge_attr, goal_u, goal_batch = saved_gnn_encoder(test_goal, None)

        
        sim_label = torch.ones(1).squeeze().to(device)
        L_dynamics = loss_cos_sim(state_u.squeeze(), goal_u.squeeze(), sim_label)

        test_loss = L_dynamics.item()


        print("#########################################")
        print("L_dynamics:", L_dynamics.item())
        print("\n")

        if check_each:
            input()


def inference_dynamics_test(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num=None, check_each = False):
    
    model_param = [data_dir, hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(), "result", "dynamics", "_".join(list(map(str, model_param))))

    if infer_num is not None:
        dynamics_model_name = 'DynamicsModel_{}.pt'.format(infer_num)
        gnn_encoder_name = 'GNNEncoder_{}.pt'.format(infer_num)
    else:
        dynamics_model_name = 'DynamicsModel_best.pt'
        gnn_encoder_name = 'GNNEncoder_best.pt'


    with open(os.path.join(model_path, "dynamics_loss_data"), "rb") as file:
        loss_data = pickle.load(file)
    
    epoch_list = loss_data["epoch"]
    
    #Loss Plot
    plt.figure(0)
    plt.figure(figsize=(5, 5))
    plt.suptitle('Loss')

    #Total Loss
    loss_list = []
    val_loss_list = []
    for loss in loss_data['loss']['total']['train']:
        loss_list.append(loss)
    for val_loss in loss_data['loss']['total']['val']:
        val_loss_list.append(val_loss)
    plt.subplot(1,1,1)
    plt.plot(epoch_list, loss_list, label='train')
    plt.plot(epoch_list, val_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('total')

    plt.savefig(os.path.join(model_path, "_".join(list(map(str, model_param))) +'_loss.png'))

    if show_result:
        plt.show()

    dynamics_saved_path = os.path.join(model_path, dynamics_model_name)
    saved_dynamics_model = DynamicsModel(device, hidden_dim, node_feature_size, global_dim, num_action)
    saved_dynamics_model.load_state_dict(torch.load(dynamics_saved_path))
    saved_dynamics_model.to(device)

    gnn_encoder_saved_path = os.path.join(model_path, gnn_encoder_name)
    saved_gnn_encoder = GNNEncoder_test(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim)
    saved_gnn_encoder.load_state_dict(torch.load(gnn_encoder_saved_path))
    saved_gnn_encoder.to(device)

    for param in saved_dynamics_model.parameters():
        param.requires_grad = False
    for param in saved_gnn_encoder.parameters():
        param.requires_grad = False

    data_test = DynamicsDataset(os.path.join(data_dir,'dynamics','test'))
    data_test_loader = DataLoader(data_test, 1)

    #loss_cos_sim = nn.CosineEmbeddingLoss().to(device)
    loss_bce = nn.BCEWithLogitsLoss().to(device)

    saved_dynamics_model.eval()
    saved_gnn_encoder.eval()
    for test_data in data_test_loader:
        test_state, test_goal, test_target, test_info = test_data

        test_dynamics_emb = saved_dynamics_model(test_state, test_target['action'], test_target['object'])
        
        state_x, state_edge_attr, state_u, state_batch = saved_gnn_encoder(test_state, test_dynamics_emb)

        goal_x = test_goal['x'].to(device)
        goal_edge_attr = test_goal['edge_attr'].to(device)

        sim_label = torch.ones(state_edge_attr.size(0), 1).squeeze().to(device)
        #L_dynamics = loss_cos_sim(state_edge_attr, goal_edge_attr, sim_label)
        L_dynamics = loss_bce(state_edge_attr, goal_edge_attr)
        test_loss = L_dynamics.item()


        print("#########################################")
        print("L_dynamics:", L_dynamics.item())
        print("\n")

        print("data info\n")
        print("task:", test_info["demo"])
        print("order:", test_info["order"])
        print("step: ", test_info["step"])

        print("\npredicted graph\n")
        print(torch.trunc(torch.softmax(state_edge_attr,dim=-1)))
        print("\ntarget graph\n")
        print(goal_edge_attr)

        if check_each:
            input()