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

# Basically the same as the baseline except we pass edge features 

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"


#### Action input embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#### ActionModel

class ActionModel(nn.Module):
    def __init__(self, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size):
        super(ActionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.batch_size = batch_size

        self.convs = [GINEConv(nn=nn.Sequential(nn.Linear(self.node_feature_size, self.hidden_dim),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.LeakyReLU(),),
                               edge_dim=self.edge_feature_size),
                      GINEConv(nn=nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.LeakyReLU(),
                                                nn.Linear(self.hidden_dim, self.hidden_dim),
                                                nn.LeakyReLU(),),
                               edge_dim=self.edge_feature_size)]
        
        self.action_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.num_action),
            nn.LeakyReLU(),
        )
        self.node_layers = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, input_data):
        
        x = input_data['x']
        edge_index = input_data['edge_index']
        edge_attr = input_data['edge_attr']

        x.to(device)
        edge_index.to(device)
        edge_attr.to(device)
        #print(input_data)
        #x, edge_index, edge_attr, _, _= input_data
        #print(x)
        #print(input_data['edge_index'])
        #print(edge_attr.shape)


        # Data format must match! 
        # Type 1) x : float32, edge_index : int64, edge_attr: float32  
        # print(type(x),type(edge_index),type(edge_attr))

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training = self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        

        #144X64 -> (9X16)X64 = (num_node X batch)Xhidden_dim
        x = x.reshape(self.batch_size, -1, self.hidden_dim) #batch X node X hidden
        #print(x.shape)

        action_input_emb = x.mean(axis=1)      # x feature를 합치는 과정 / 현재는 mean으로 (추후 변경 예정)
        # print("actopm=input",action_input_emb)
        #print(action_input_emb.shape) # batch X hidden
        softmax = nn.Softmax(dim=1).to(device)
        action_prob = softmax(self.action_layers(action_input_emb))
       
        # action_prob = self.action_layers(action_input_emb)
    
    
        # action_prob = nn.Softmax(self.action_layers(action_input_emb))
        
        
        each_node = []
        x = x.reshape(-1, self.batch_size, self.hidden_dim)
        for feature in x:
            #print(feature.shape) # feature size : batch X hidden

            sig = nn.Sigmoid().to(device)
            # node_scores.append(nn.Sigmoid(self.node_layers(feature)))
            each_node.append(sig(self.node_layers(feature))) #tensor는 append로 합친 후 concat을 해야 list형식의 tensor형태로 가지고 있는다.

        node_scores = torch.cat(each_node, dim=1)
        #print("node_scores", node_scores.shape)
        # print("\n[Each node]",each_node)

        return action_prob, node_scores
  

def train():
    hidden_dim = 64
    num_action = 3 # [pick, place, pour]
    node_feature_size = 6 #노드 feature 크기
    edge_feature_size = 13 # 노드 사이의 relation 종류 개수 [on_right,on_left, in_right, in_left, attach, in-grasp]
    batch_size = 16

    model = ActionModel(hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size)
    model.to(device)

    total_dataset = StackingDataset('stacking_dataset')
    train_dataset, test_dataset = random_split(total_dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    #train_loader = DataLoader(train_dataset, shuffle=True)
    #test_loader = DataLoader(test_dataset, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0005)
    loss = nn.CrossEntropyLoss().to(device)

    best_loss = 10000
    
    loss_data = {"epoch":[],
                 "test_loss":[]}

    #train
    for epoch in range(1000):
        print("#############################")
        print("epoch number {}".format(epoch+1))
        model.train()

        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(train_loader):
            input, target = data
            #input = input.to(device)
            #target = target.to(device)


            optimizer.zero_grad()
            
            pred_action_prob, pred_node_scores = model(input)
            target_action_prob, target_node_scores = target['action'], target['object']
            target_action_prob.to(device)
            target_node_scores.to(device)


            # CrossEntropyLoss 1) Input (N,C) C= number of classes / Target N where each value is 0

            #Action loss
            # target_action_prob = torch.empty(3).random_(2) #세 개 중에 하나만 1로 설정해야함
            #print("pred_action",pred_action_prob.dtype)
            #print("target_action",target_action_prob.dtype)
            L_action = loss(pred_action_prob, target_action_prob)

            #Nodescore loss
            # target_node_scores = torch.empty(8).random_(2)
            #print("pred_nodescore",pred_node_scores.dtype)
            #print("target_nodescore",target_node_scores.dtype)
            L_nodescore = loss(pred_node_scores, target_node_scores)

            # Total loss
            L_total = L_action + L_nodescore
            # print(L_total)

            L_total.backward()

            optimizer.step()
            running_loss += L_total.item()

            if (i+1) % batch_size == 0:
                last_loss = running_loss / batch_size
                print("batch: {} loss: {}".format(i, last_loss))
                running_loss = 0.0
        
        test_running_loss = 0.0
        model.eval()
        for i, data in enumerate(test_loader):
            test_input, test_target = data
            #test_input.to(device)
            #test_target.to(device)

            test_pred_action_prob, test_pred_node_scores = model(test_input)
            test_target_action_prob, test_target_node_scores = test_target['action'], test_target['object']

            test_L_action = loss(test_pred_action_prob, test_target_action_prob)
            test_L_nodescore = loss(test_pred_node_scores, test_target_node_scores)

            test_L_total = test_L_action + test_L_nodescore
            
            test_running_loss += test_L_total
        
        test_avg_loss = test_running_loss / batch_size
        print("Loss train: {} test: {}".format(last_loss, test_avg_loss))

        loss_data['epoch'].append(epoch)
        loss_data['test_loss'].append(test_avg_loss)

        if test_avg_loss < best_loss:
            best_loss = test_avg_loss
            model_path = './stacking_model/stacking_model_{}'.format(epoch)
            torch.save(model.state_dict(), model_path)
    
    #save loss record
    file_path = "./stacking_model/loss_data"
    with open(file_path, "wb") as outfile:
        pickle.dump(loss_data, outfile)



train()
