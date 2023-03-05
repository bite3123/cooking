from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import  random_split
import torch.optim as optim
from GraphPlanningDataset import GraphPlanningDataset
from ActionModel import *
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

    saved_model = ActionModel_act_only(device, hidden_dim, num_action, node_feature_size, edge_feature_size)
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
        print("pred_action_prob:", pred_action_prob)
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


def inference(device, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size, lr, num_epoch, data_dir, show_result, infer_num=None, check_each = False):
    
    model_param = [data_dir, hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(), "result", "_".join(list(map(str, model_param))))

    if infer_num is not None:
        model_name = 'GP_model_{}.pt'.format(infer_num)
    else:
        model_name = 'GP_model_best.pt'


    with open(os.path.join(model_path, "loss_data"), "rb") as file:
        loss_data = pickle.load(file)
    
    epoch_list = loss_data["epoch"]
    
    #Loss Plot
    plt.figure(0)
    plt.figure(figsize=(15, 5))
    plt.suptitle('Loss')

    #Total Loss
    loss_list = []
    val_loss_list = []
    for loss in loss_data['loss']['total']['train']:
        loss_list.append(loss)
    for val_loss in loss_data['loss']['total']['val']:
        val_loss_list.append(val_loss)
    plt.subplot(1,3,1)
    plt.plot(epoch_list, loss_list, label='train')
    plt.plot(epoch_list, val_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('total')

    #Action Loss
    act_loss_list = []
    val_act_loss_list = []
    for loss in loss_data['loss']['action']['train']:
        act_loss_list.append(loss)
    for val_loss in loss_data['loss']['action']['val']:
        val_act_loss_list.append(val_loss)
    plt.subplot(1,3,2)
    plt.plot(epoch_list, act_loss_list, label='train')
    plt.plot(epoch_list, val_act_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('action')

    #Object Loss
    obj_loss_list = []
    val_obj_loss_list = []
    for loss in loss_data['loss']['object']['train']:
        obj_loss_list.append(loss)
    for val_loss in loss_data['loss']['object']['val']:
        val_obj_loss_list.append(val_loss)
    plt.subplot(1,3,3)
    plt.plot(epoch_list, obj_loss_list, label='train')
    plt.plot(epoch_list, val_obj_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('object')

    plt.savefig(os.path.join(model_path, "_".join(list(map(str, model_param))) +'_loss.png'))

    #Accuracy Plot
    plt.figure(1)
    plt.figure(figsize=(10, 5))
    plt.suptitle('Accuracy')

    #Action Accuracy
    act_acc_list = []
    val_act_acc_list = []
    for acc in loss_data['acc']['action']['train']:
        act_acc_list.append(acc)
    for val_acc in loss_data['acc']['action']['val']:
        val_act_acc_list.append(val_acc)
    plt.subplot(1,2,1)
    plt.plot(epoch_list, act_acc_list, label='train')
    plt.plot(epoch_list, val_act_acc_list , label='val')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.title('action')

    #Object Accuracy
    obj_acc_list = []
    val_obj_acc_list = []
    for acc in loss_data['acc']['object']['train']:
        obj_acc_list.append(acc)
    for val_acc in loss_data['acc']['object']['val']:
        val_obj_acc_list.append(val_acc)
    plt.subplot(1,2,2)
    plt.plot(epoch_list, obj_acc_list, label='train')
    plt.plot(epoch_list, val_obj_acc_list , label='val')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.title('object')

    plt.savefig(os.path.join(model_path, "_".join(list(map(str, model_param))) +'_acc.png'))
    if show_result:
        plt.show()

    saved_path = os.path.join(model_path, model_name)

    saved_model = ActionModel(device, hidden_dim, num_action, node_feature_size, edge_feature_size)
    saved_model.load_state_dict(torch.load(saved_path))
    saved_model.to(device)

    for param in saved_model.parameters():
        param.requires_grad = False

    data_test = GraphPlanningDataset(os.path.join(data_dir,'test'))
    data_test_loader = DataLoader(data_test, 1)

    loss_ce_action = nn.CrossEntropyLoss().to(device)
    loss_ce_object = nn.CrossEntropyLoss().to(device)

    act_num_total = [0, 0, 0]
    act_num_acc = [0, 0, 0]

    obj_num_acc = 0

    saved_model.eval()
    for test_data in data_test_loader:
        test_input, test_target, test_info = test_data
        pred_action_prob, pred_object_prob = saved_model(test_input)

        target_action_prob, target_node_scores = test_target['action'].to(device), test_target['object'].to(device)
        
        test_act_label = torch.argmax(target_action_prob, dim=1).to(device)
        test_obj_label = torch.argmax(target_node_scores, dim=1).to(device)

        L_action = loss_ce_action(pred_action_prob, test_act_label)
        L_object = loss_ce_object(pred_object_prob, test_obj_label)
        L_total = L_action + L_object


        print("#########################################")
        print("data info:")
        print("demo type:", test_info['demo'])
        print("order:", test_info['order'])
        print("step:", test_info['step'].item())
        print("--------------------------------")
        print("Loss:")
        print("--------------------------------")
        print("L_total:", L_total.item())
        print("\n")
        print("pred_action_score:", F.softmax(pred_action_prob, dim=-1))
        print("target_action_prob:",target_action_prob)
        print("L_action:", L_action.item())

        act_num_total[test_act_label.item()] += 1
        print("Action Prediction Result:")
        if torch.argmax(pred_action_prob, dim=1) == test_act_label:
            act_num_acc[test_act_label.item()] += 1
            print("Success!")
        else:
            print("Failed TT")

        print("\n")
        print("pred_object_score:", F.softmax(pred_object_prob, dim=-1))
        print("target_object_prob:",target_node_scores)
        print("L_object:", L_object.item())
        
        print("Object Prediction Result:")
        if torch.argmax(pred_object_prob, dim=1) == test_obj_label:
            obj_num_acc += 1
            print("Success!")
        else:
            print("Failed TT")

        if check_each:
            input()
        
    print("------------------------")
    print("Accuracy:")
    print("------------------------")
    print("Pick Result: {}/{} corrected".format(act_num_acc[0], act_num_total[0]))
    print("Pick Acc: {:01.4f}".format(act_num_acc[0]/ act_num_total[0]))
    print("------------------------")
    print("Place Result: {}/{} corrected".format(act_num_acc[1], act_num_total[1]))
    print("Place Acc: {:01.4f}".format(act_num_acc[1]/ act_num_total[1]))
    print("------------------------")
    print("Pour Result: {}/{} corrected".format(act_num_acc[2], act_num_total[2]))
    print("Pour Acc: {:01.4f}".format(act_num_acc[2]/ act_num_total[2]))
    print("------------------------")
    print("------------------------")
    print("Action Result: {}/{} corrected".format(sum(act_num_acc), sum(act_num_total)))
    print("Action Acc: {:01.4f}".format(sum(act_num_acc)/sum(act_num_total)))

    print("------------------------")
    print("------------------------")
    print("Target Object Result: {}/{} corrected".format(obj_num_acc, sum(act_num_total)))
    print("Target Object Acc: {:01.4f}".format(obj_num_acc/sum(act_num_total)))