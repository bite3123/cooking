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
from ActionModel import ActionModel
import os

def train_act_only(device, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size, lr, num_epoch, data_dir):
    model = ActionModel(device, hidden_dim, num_action, node_feature_size, edge_feature_size)

    model.to(device)

    model_name = [data_dir, hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(), "result", "_".join(list(map(str, model_name))))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_dataset = GraphPlanningDataset(os.path.join(data_dir,'train'))
    val_dataset = GraphPlanningDataset(os.path.join(data_dir,'val'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    action_weights = torch.tensor([1080/(1080+1080+120),1080/(1080+1080+120),120/(1080+1080+120)])
    loss_ce = nn.CrossEntropyLoss(weight=action_weights).to(device)
    #loss_bce = nn.BCEWithLogitsLoss().to(device)

    for param in model.parameters():
        param.requires_grad = True

    best_loss = 10000

    loss_data = {"epoch":[],
                "loss":{"train":[],
                        "val":[]},
                "acc":{"train":[],
                        "val":[]}}

    #train
    for epoch in range(num_epoch):
        print("#############################")
        print("epoch number {}".format(epoch+1))
        model.train()

        running_loss = 0.0
        last_loss = 0.0
        num_correct = 0
        num_total = 0

        for i, data in enumerate(train_loader):
            input, target = data

            pred_action_prob= model(input)

            target_action_prob, target_node_scores = target['action'], target['object']
            target_action_prob.to(device)
            target_node_scores.to(device)

            act_label = torch.argmax(target_action_prob, dim=1).to(device)

            L_action = loss_ce(pred_action_prob,act_label)
            
            L_action.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += L_action.item()
            last_loss = running_loss/(i+1)

            num_correct += torch.sum(torch.argmax(pred_action_prob, dim=-1)==act_label)
            num_total += act_label.size(dim=0)

        val_running_loss = 0.0
        val_avg_loss = 0.0

        val_num_correct = 0
        val_num_total = 0

        model.eval()
        
        for i, data in enumerate(val_loader):
            val_input, val_target= data
            
            val_pred_action_prob = model(val_input)
            val_target_action_prob, val_target_node_scores = val_target['action'], val_target['object']
            
            val_target_action_prob.to(device)
            val_target_node_scores.to(device)

            val_act_label = torch.argmax(val_target_action_prob, dim=1).to(device)
            val_L_action = loss_ce(val_pred_action_prob, val_act_label)
                    
            val_running_loss += val_L_action.item()
            val_avg_loss = val_running_loss / (i+1)

            val_num_correct += torch.sum(torch.argmax(val_pred_action_prob, dim=-1)==val_act_label)
            val_num_total += val_act_label.size(dim=0)

        acc = num_correct.item()/num_total
        val_acc = val_num_correct.item()/val_num_total
        print("Acc\ttrain:{:01.4f}\tval:{:01.4f}".format(acc, val_acc))
        print("Loss\ttrain:{:01.4f}\tval:{:01.4f}".format(last_loss, val_avg_loss))

        loss_data['epoch'].append(epoch)
        loss_data['loss']['train'].append(last_loss)
        loss_data['loss']['val'].append(val_avg_loss)
        loss_data['acc']['train'].append(acc)
        loss_data['acc']['val'].append(val_acc)

        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            torch.save(model.state_dict(), model_path + '/GP_model_{}.pth'.format(epoch))
            torch.save(model.state_dict(), model_path + '/GP_model_best.pth')
        

    #save loss record
    file_path = os.path.join(model_path, 'loss_data')
    with open(file_path, "wb") as outfile:
        pickle.dump(loss_data, outfile)
