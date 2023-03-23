from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from GraphPlanningDataset import *
import pickle
from ActionModel import *
import os

def train_action(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir):
    model = ActionModel(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim)

    model.to(device)

    model_name = [data_dir, hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(), "result", "action","_".join(list(map(str, model_name))))
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_dataset = ActionDataset(os.path.join(data_dir, 'action','train'))
    val_dataset = ActionDataset(os.path.join(data_dir,'action','val'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    #action_dist = [1200, 1200, 120]
    #action_weights = torch.tensor([action_dist[0]/sum(action_dist),action_dist[1]/sum(action_dist),action_dist[2]/sum(action_dist)])
    
    #loss_ce_action = nn.CrossEntropyLoss(weight=action_weights).to(device)
    loss_ce_action = nn.CrossEntropyLoss().to(device)
    loss_ce_object = nn.CrossEntropyLoss().to(device)

    for param in model.parameters():
        param.requires_grad = True

    best_loss = 10000
    
    loss_data = {"epoch":[],
                 "loss":{"total":{"train":[],
                                  "val":[]},
                         "action":{"train":[],
                                   "val":[]},
                         "object":{"train":[],
                                   "val":[]}},
                 "acc":{"action":{"train":[],
                                   "val":[]},
                         "object":{"train":[],
                                   "val":[]}}}

    #train
    for epoch in range(num_epoch):
        print("#############################")
        print("epoch number {}".format(epoch+1))
        model.train()

        running_loss = 0.0
        last_loss = 0.0
        act_running_loss = 0.0
        act_last_loss = 0.0
        obj_running_loss = 0.0
        obj_last_loss = 0.0

        num_act_correct = 0
        num_act_total = 0
        num_obj_correct = 0
        num_obj_total = 0

        for i, data in enumerate(train_loader):
            input, target, info = data

            pred_action_prob, pred_object_prob = model(input)

            target_action_prob, target_node_scores = target['action'], target['object']
            target_action_prob.to(device)
            target_node_scores.to(device)

            act_label = torch.argmax(target_action_prob, dim=1).to(device)
            obj_label = torch.argmax(target_node_scores, dim=1).to(device)

            L_action = loss_ce_action(pred_action_prob, act_label)
            L_object = loss_ce_object(pred_object_prob, obj_label)
            
            L_total = L_action + L_object
            L_total.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += L_total.item()
            last_loss = running_loss/(i+1)
            act_running_loss += L_action.item()
            act_last_loss = act_running_loss/(i+1)
            obj_running_loss += L_object.item()
            obj_last_loss = obj_running_loss/(i+1)

            num_act_correct += torch.sum(torch.argmax(pred_action_prob, dim=-1)==act_label)
            num_act_total += act_label.size(dim=0)
            num_obj_correct += torch.sum(torch.argmax(pred_object_prob, dim=-1)==obj_label)
            num_obj_total += obj_label.size(dim=0)

        val_running_loss = 0.0
        val_last_loss = 0.0
        val_act_running_loss = 0.0
        val_act_last_loss = 0.0
        val_obj_running_loss = 0.0
        val_obj_last_loss = 0.0

        val_num_act_correct = 0
        val_num_act_total = 0
        val_num_obj_correct = 0
        val_num_obj_total = 0

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_input, val_target, _ = data
                
                val_pred_action_prob, val_pred_object_prob = model(val_input)
                val_target_action_prob, val_target_node_scores = val_target['action'], val_target['object']
                
                val_target_action_prob.to(device)
                val_target_node_scores.to(device)

                val_act_label = torch.argmax(val_target_action_prob, dim=1).to(device)
                val_L_action = loss_ce_action(val_pred_action_prob, val_act_label)

                val_obj_label = torch.argmax(val_target_node_scores, dim=1).to(device)
                val_L_object = loss_ce_object(val_pred_object_prob, val_obj_label)

                val_L_total = val_L_action + val_L_object
                        
                val_running_loss += val_L_total.item()
                val_last_loss = val_running_loss / (i+1)


                val_act_running_loss += val_L_action.item()
                val_act_last_loss = val_act_running_loss/(i+1)

                val_obj_running_loss += val_L_object.item()
                val_obj_last_loss = val_obj_running_loss/(i+1)

                val_num_act_correct += torch.sum(torch.argmax(val_pred_action_prob, dim=-1)==val_act_label)
                val_num_act_total += val_act_label.size(dim=0)

                val_num_obj_correct += torch.sum(torch.argmax(val_pred_object_prob, dim=-1)==val_obj_label)
                val_num_obj_total += val_obj_label.size(dim=0)

        act_acc = num_act_correct.item()/num_act_total
        obj_acc = num_obj_correct.item()/num_obj_total
        
        val_act_acc = val_num_act_correct.item()/val_num_act_total
        val_obj_acc = val_num_obj_correct.item()/val_num_obj_total
        
        print("Action Acc\ttrain:{:01.4f}\tval:{:01.4f}".format(act_acc, val_act_acc))
        print("Object Acc\ttrain:{:01.4f}\tval:{:01.4f}".format(obj_acc, val_obj_acc))  

        print("\nTotal Loss\ttrain:{:01.4f}\tval:{:01.4f}".format(last_loss, val_last_loss))
        print("Action Loss\ttrain:{:01.4f}\tval:{:01.4f}".format(act_last_loss, val_act_last_loss))
        print("Object Loss\ttrain:{:01.4f}\tval:{:01.4f}".format(obj_last_loss, val_obj_last_loss))

        loss_data['epoch'].append(epoch)
        
        loss_data['acc']['action']['train'].append(act_acc)
        loss_data['acc']['action']['val'].append(val_act_acc)
        loss_data['acc']['object']['train'].append(obj_acc)
        loss_data['acc']['object']['val'].append(val_obj_acc)

        loss_data['loss']['total']['train'].append(last_loss)
        loss_data['loss']['total']['val'].append(val_last_loss)
        loss_data['loss']['action']['train'].append(act_last_loss)
        loss_data['loss']['action']['val'].append(val_act_last_loss)
        loss_data['loss']['object']['train'].append(obj_last_loss)
        loss_data['loss']['object']['val'].append(val_obj_last_loss)

        if val_last_loss < best_loss:
            best_loss = val_last_loss
            torch.save(model.state_dict(), model_path + '/GP_model_{}.pt'.format(epoch))
            torch.save(model.state_dict(), model_path + '/GP_model_best.pt')


        #save loss record
        file_path = os.path.join(model_path, 'loss_data')
        with open(file_path, "wb") as outfile:
            pickle.dump(loss_data, outfile)
