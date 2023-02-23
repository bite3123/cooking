from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import  random_split
import torch.optim as optim
from StackingDataset import StackingDataset
from ActionModel import ActionModel
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"

hidden_dim = 32
num_action = 3 # [pick, place, pour]
node_feature_size = 6 #노드 feature 크기
edge_feature_size = 26 # 노드 사이의 relation 종류 개수 [on_right,on_left, in_right, in_left, attach, in-grasp]
batch_size = 1

saved_path = "./stacking_model/stacking_model_25.pth"

saved_model = ActionModel(hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size)
saved_model.load_state_dict(torch.load(saved_path))
data_test = StackingDataset('stacking_dataset_test')

data_test_loader = DataLoader(data_test, batch_size)

for test_input, test_target in data_test_loader:
    #test = data_test_loader
    #test_input, test_target = test
    #test_input, test_target = data_test_loader.__getitem__(0)
    print("#########################################")
    print("test data:", test_input, test_target)

    pred_action_prob, pred_node_scores = saved_model(test_input, test_target)
    print("pred_action_prob:", pred_action_prob)
    print("pred_node_scores:", pred_node_scores)

    target_action_prob, target_node_scores = test_target['action'], test_target['object']

    print("target_action_prob:",target_action_prob)
    print("target_node_scores:", target_node_scores)


    loss = nn.CrossEntropyLoss().to(device)

    L_action = loss(pred_action_prob, target_action_prob)
    print("L_action:", L_action)
    L_nodescore = loss(pred_node_scores, target_node_scores)
    print("L_nodescore", L_nodescore)
    L_total = L_action + L_nodescore
    print("L_total", L_total)

    input()