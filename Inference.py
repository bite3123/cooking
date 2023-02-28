from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import  random_split
import torch.optim as optim
from GraphPlanningDataset import GraphPlanningDataset
from ActionModel import ActionModel
import matplotlib.pyplot as plt
import pickle
import os

def inference_act_only(device, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size, lr, data_dir, show_result, infer_num=None):
    
    model_path = os.path.join(os.getcwd(), 'model', data_dir + '_' + str(batch_size) + '_' + str(lr))

    if infer_num is not None:
        model_name = 'GP_model_{}.pth'.format(infer_num)
    else:
        model_name = 'GP_model_best.pth'


    
    if show_result:
        with open(os.path.join(model_path, "loss_data"), "rb") as file:
            loss_data = pickle.load(file)

        loss_list = []
        val_loss_list = []
        for loss in loss_data['loss']['train']:
            loss_list.append(loss)
        for val_loss in loss_data['loss']['val']:
            val_loss_list.append(val_loss)


        plt.subplot(1,2,1)
        plt.plot(loss_data["epoch"], loss_list, label='train')
        plt.plot(loss_data["epoch"], val_loss_list , label='val')
        plt.legend(loc='upper right')
        plt.title('Loss')

        acc_list = []
        val_acc_list = []
        for acc in loss_data['acc']['train']:
            acc_list.append(acc)
        for val_acc in loss_data['acc']['val']:
            val_acc_list.append(val_acc)

        plt.subplot(1,2,2)
        plt.plot(loss_data['epoch'], acc_list, label='train')
        plt.plot(loss_data['epoch'], val_acc_list, label='val')
        plt.legend(loc='upper left')
        plt.title('Accuracy')
        
        plt.savefig(os.path.join(model_path, data_dir + '_' + str(batch_size) + '_' + str(lr) +'.png'))
        plt.show()
        


    saved_path = os.path.join(model_path, model_name)

    saved_model = ActionModel(hidden_dim, num_action, node_feature_size, edge_feature_size)
    saved_model.load_state_dict(torch.load(saved_path))

    data_test = GraphPlanningDataset(os.path.join(data_dir,'test'))
    data_test_loader = DataLoader(data_test, 1)

    saved_model.eval()
    for test_input, test_target in data_test_loader:
        print("#########################################")
        pred_action_prob = saved_model(test_input)
        print("pred_action_score:", F.softmax(pred_action_prob, dim=-1))

        target_action_prob, target_node_scores = test_target['action'], test_target['object']

        print("target_action_prob:\n",target_action_prob)
        #print("target_node_scores:\n", target_node_scores)


        loss_bce = nn.BCEWithLogitsLoss().to(device)

        L_action_bce = loss_bce(pred_action_prob, target_action_prob)
        print("L_bce:", L_action_bce)

        test_act_label = torch.argmax(target_action_prob, dim=1)
        loss_ce = nn.CrossEntropyLoss().to(device)
        L_action_ce = loss_ce(pred_action_prob, test_act_label)
        print("L_CE:", L_action_ce)

        print("Prediction Result:")
        if torch.argmax(pred_action_prob, dim=1) == test_act_label:
            print("Success!")
        else:
            print("Failed TT")


        input()