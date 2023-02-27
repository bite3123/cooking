from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import  random_split
import torch.optim as optim
from StackingDataset import StackingDataset
from ActionModel import ActionModel
import matplotlib.pyplot as plt
import pickle
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

hidden_dim = 256
num_action = 3 # [pick, place, pour]
node_feature_size = 6 #노드 feature 크기
edge_feature_size = 14 # 노드 사이의 relation 종류 개수 [on_right,on_left, in_right, in_left, attach, in-grasp]
batch_size = 1
'''
with open("./stacking_model_nopos_nofc_act_adam_CE/loss_data", "rb") as file:
    loss_data = pickle.load(file)
#print((loss_data["epoch"], loss_data["test_loss"]))
t_list = []
train_list = []
for t in loss_data["test_loss"]:
    t_list.append(t.detach().numpy())
for train_loss in loss_data["train_loss"]:
    train_list.append(train_loss)
plt.plot(loss_data["epoch"], t_list)
plt.plot(loss_data["epoch"], train_list)
plt.show()
'''
saved_path = "./stacking_model_nopos_nofc_act_adam_CE/stacking_model_1.pth"

saved_model = ActionModel(hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size)
saved_model.load_state_dict(torch.load(saved_path))
data_test = StackingDataset('stacking_dataset_nopos_nofc')

data_test_loader = DataLoader(data_test, batch_size)
saved_model.eval()
for test_input, test_target, test_key_node in data_test_loader:
    #test = data_test_loader
    #test_input, test_target = test
    #test_input, test_target = data_test_loader.__getitem__(0)
    print("#########################################")
    print("test data:", test_input, test_target)

    pred_action_prob = saved_model(test_input, test_key_node)
    print("pred_action_prob:", pred_action_prob)

    target_action_prob, target_node_scores = test_target['action'], test_target['object']

    print("target_action_prob:",target_action_prob)
    print("target_node_scores:", target_node_scores)


    loss = nn.BCEWithLogitsLoss().to(device)

    L_action = loss(pred_action_prob, target_action_prob)
    print("L_action:", L_action)

    loss_2 = nn.BCELoss().to(device)
    sig = torch.sigmoid(pred_action_prob)
    print("sig:", sig)
    print("target:", target_action_prob)
    L_2 = loss_2(sig, target_action_prob)
    print("L_BCE:", L_2)

    temp = torch.argmax(target_action_prob, dim=1)
    loss_CE = nn.CrossEntropyLoss().to(device)
    L_CE = loss_CE(pred_action_prob, temp)
    print("CE:", L_CE)

    print("Prediction Result:")
    if torch.argmax(pred_action_prob, dim=1) == temp:
        print("Success!")
    else:
        print("Failed TT")


    input()


def inference(data_test):
    device = "cpu"

    hidden_dim = 8
    num_action = 3 # [pick, place, pour]
    node_feature_size = 6 #노드 feature 크기
    edge_feature_size = 14 # 노드 사이의 relation 종류 개수 [on_right,on_left, in_right, in_left, attach, in-grasp]
    batch_size = 1

    with open("./stacking_model_nopos_nofc_act_adam_BCE/loss_data", "rb") as file:
        loss_data = pickle.load(file)
    #print((loss_data["epoch"], loss_data["test_loss"]))
    t_list = []
    for t in loss_data["test_loss"]:
        t_list.append(t.detach().numpy())
    plt.plot(loss_data["epoch"], t_list)
    plt.show()

    saved_path = "./stacking_model_nopos_nofc_act_adam_BCE/stacking_model_199.pth"

    saved_model = ActionModel(hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size)
    saved_model.load_state_dict(torch.load(saved_path))
    #data_test = StackingDataset('stacking_dataset_nopos_nofc')

    data_test_loader = DataLoader(data_test, batch_size)

    for test_input, test_target in data_test_loader:
        #test = data_test_loader
        #test_input, test_target = test
        #test_input, test_target = data_test_loader.__getitem__(0)
        print("#########################################")
        print("test data:", test_input, test_target)

        pred_action_prob = saved_model(test_input, test_target)
        print("pred_action_prob:", pred_action_prob)

        target_action_prob, target_node_scores = test_target['action'], test_target['object']

        print("target_action_prob:",target_action_prob)
        print("target_node_scores:", target_node_scores)


        loss = nn.BCEWithLogitsLoss().to(device)

        L_action = loss(pred_action_prob, target_action_prob)
        print("L_action:", L_action)

        input()