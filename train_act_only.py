from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import  random_split
import torch.optim as optim
from StackingDataset import StackingDataset
import pickle
from ActionModel import ActionModel

# Basically the same as the baseline except we pass edge features 

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"


hidden_dim = 128
num_action = 3 # [pick, place, pour]
node_feature_size = 6 #노드 feature 크기
edge_feature_size = 14 # 노드 사이의 relation 종류 개수 [on_right,on_left, in_right, in_left, attach, in-grasp]
batch_size = 64

model = ActionModel(hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size)
model.to(device)

total_dataset = StackingDataset('stacking_dataset_nopos_nofc')
train_dataset, test_dataset = random_split(total_dataset, [0.8, 0.2])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#train_loader = DataLoader(train_dataset, shuffle=True)
#test_loader = DataLoader(test_dataset, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
loss = nn.CrossEntropyLoss().to(device)
#loss_test = nn.BCEWithLogitsLoss().to(device)
for param in model.parameters():
    param.requires_grad = True

best_loss = 10000

loss_data = {"epoch":[],
             "train_loss":[],
             "test_loss":[]}

#train
for epoch in range(200):
    print("#############################")
    print("epoch number {}".format(epoch+1))
    model.train()

    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(train_loader):
        input, target, key_node = data
        #print(key_node)       
        #input = input.to(device)
        #target = target.to(device)


        
        #pred_action_prob, pred_node_scores = model(input, target)
        pred_action_prob= model(input, key_node)
        target_action_prob, target_node_scores = target['action'], target['object']
        target_action_prob.to(device)
        target_node_scores.to(device)

        #temp = (target_action_prob==1).nonzero()[:,-1]
        temp = torch.argmax(target_action_prob, dim=1)
        #print(temp.shape,"\n", temp)
        #print(pred_action_prob.shape,"\n",pred_action_prob)


        # CrossEntropyLoss 1) Input (N,C) C= number of classes / Target N where each value is 0

        #Action loss
        # target_action_prob = torch.empty(3).random_(2) #세 개 중에 하나만 1로 설정해야함
        #print("pred_action",pred_action_prob.dtype)
        #print("target_action",target_action_prob.dtype)
        L_action = loss(pred_action_prob,temp)
        #print(L_action.item())
        #print(pred_action_prob)
        #print(target_action_prob)
        #print(temp)
        #L_action = loss_test(pred_action_prob, target_action_prob)
        #Nodescore loss
        # target_node_scores = torch.empty(8).random_(2)
        #print("pred_nodescore",pred_node_scores.dtype)
        #print("target_nodescore",target_node_scores.dtype)
        #L_nodescore = loss(pred_node_scores, target_node_scores)

        # Total loss
        #L_total = L_action + L_nodescore
        # print(L_total)

        #L_total.backward()

        optimizer.zero_grad()
        L_action.backward()
        optimizer.step()
        running_loss += L_action.item()

        if (i) % batch_size == 0:
            last_loss = running_loss / batch_size
            print("batch: {} loss: {}".format(i, last_loss))
            running_loss = 0.0
    
    test_running_loss = 0.0
    model.eval()
    for i, data in enumerate(test_loader):
        test_input, test_target , test_key_node = data
        #test_input.to(device)
        #test_target.to(device)

        test_pred_action_prob = model(test_input, test_key_node)
        test_target_action_prob, test_target_node_scores = test_target['action'], test_target['object']

        temp = torch.argmax(test_target_action_prob, dim=1)
        test_L_action = loss(test_pred_action_prob, temp)
        #test_L_nodescore = loss(test_pred_node_scores, test_target_node_scores)

        #test_L_total = test_L_action + test_L_nodescore
        
        test_running_loss += test_L_action
    
    test_avg_loss = test_running_loss / batch_size
    print("Loss train: {} test: {}".format(last_loss, test_avg_loss))

    loss_data['epoch'].append(epoch)
    loss_data['train_loss'].append(last_loss)
    loss_data['test_loss'].append(test_avg_loss)

    if test_avg_loss < best_loss:
        best_loss = test_avg_loss
        model_path = './stacking_model_nopos_nofc_act_adam_CE/stacking_model_{}.pth'.format(epoch)
        torch.save(model.state_dict(), model_path)
    

#save loss record
file_path = "./stacking_model_nopos_nofc_act_adam_CE/loss_data"
with open(file_path, "wb") as outfile:
    pickle.dump(loss_data, outfile)
