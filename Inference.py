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

def inference_act_only(device, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size, lr, num_epoch, data_dir, show_result, infer_num=None, check_each = False):
    
    model_param = [data_dir, hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(), "result", "_".join(list(map(str, model_param))))

    if infer_num is not None:
        model_name = 'GP_model_{}.pt'.format(infer_num)
    else:
        model_name = 'GP_model_best.pt'


    with open(os.path.join(model_path, "loss_data"), "rb") as file:
        loss_data = pickle.load(file)

    loss_list = []
    val_loss_list = []
    for loss in loss_data['loss']['train']:
        loss_list.append(loss)
    for val_loss in loss_data['loss']['val']:
        val_loss_list.append(val_loss)

    plt.figure(figsize=(10, 5))

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
    plt.ylim([0, 1])
    plt.legend(loc='upper left')
    plt.title('Accuracy')
    
    plt.savefig(os.path.join(model_path, "_".join(list(map(str, model_param))) +'.png'))
    if show_result:
        plt.show()
        


    saved_path = os.path.join(model_path, model_name)

    saved_model = ActionModel(device, hidden_dim, num_action, node_feature_size, edge_feature_size)
    saved_model.load_state_dict(torch.load(saved_path))
    saved_model.to(device)

    data_test = GraphPlanningDataset(os.path.join(data_dir,'test'))
    data_test_loader = DataLoader(data_test, 1)

    loss_bce = nn.BCEWithLogitsLoss().to(device)
    loss_ce = nn.CrossEntropyLoss().to(device)

    num_total = [0, 0, 0]
    num_acc = [0, 0, 0]
    saved_model.eval()
    for test_input, test_target in data_test_loader:
        pred_action_prob = saved_model(test_input)
        target_action_prob, target_node_scores = test_target['action'].to(device), test_target['object'].to(device)

        L_action_bce = loss_bce(pred_action_prob, target_action_prob)
        
        test_act_label = torch.argmax(target_action_prob, dim=1).to(device)
        L_action_ce = loss_ce(pred_action_prob, test_act_label)


        print("#########################################")
        print()
        #print("test data {}".format(num_total))
        print("pred_action_score:", F.softmax(pred_action_prob, dim=-1))
        print("target_action_prob:\n",target_action_prob)
        #print("target_node_scores:\n", target_node_scores)
    
        print("L_bce:", L_action_bce.item())     
        print("L_ce:", L_action_ce.item())

        num_total[test_act_label.item()] += 1

        print("Prediction Result:")
        if torch.argmax(pred_action_prob, dim=1) == test_act_label:
            num_acc[test_act_label.item()] += 1
            print("Success!")
        else:
            print("Failed TT")


        if check_each:
            input()
                
    
    
    if check_each is False:

        print("------------------------")
        print("Pick Result: {}/{} corrected".format(num_acc[0], num_total[0]))
        print("Pick Acc: {:01.4f}".format(num_acc[0]/ num_total[0]))
        print("------------------------")
        print("Place Result: {}/{} corrected".format(num_acc[1], num_total[1]))
        print("Place Acc: {:01.4f}".format(num_acc[1]/ num_total[1]))
        print("------------------------")
        print("Pour Result: {}/{} corrected".format(num_acc[2], num_total[2]))
        print("Pour Acc: {:01.4f}".format(num_acc[2]/ num_total[2]))
        print("------------------------")
        print("------------------------")
        print("Test Result: {}/{} corrected".format(sum(num_acc), sum(num_total)))
        print("Test Acc: {:01.4f}".format(sum(num_acc)/sum(num_total)))
