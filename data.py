import os
import torch
import pandas as pd
import torch_geometric
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Dataset

## Check dataframe
class LoadDataFrame(Dataset):
    ## Load DataFrame
    def __init__(self, DATA_PATH):
        data = pd.read_csv(DATA_PATH)
        self.data = data
        # print(data.head)

    ## Example of general information about the dataset
    def general_info(self):
        print(self.data.shape)
        print(self.data["Position_t4"].value_counts()) # value) 6 zeros 2 ones

    ## Quick check with versions
    def checkversion(self):
        print(f"Torch version: {torch.__version__}") 
        print(f"Cuda available: {torch.cuda.is_available()}") 
        print(f"Torch geometric version: {torch_geometric.__version__}") 



## Dataset
class MakeDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

        # Search path
        search_path = os.path.join(os.getcwd(), self.root_path)
        self.search_path = search_path
        print(search_path)


    # Creating sample data
    def rand_sample(self, file_name, folder_name, save_dir, n):

        # Read node feature datasets
        node_path = os.path.join(self.search_path, folder_name, file_name)
        nf_csv = pd.read_csv(node_path, index_col=0)  # index_col drops 'Unnamed' column

        # Make random samples with torch
        for i in range(n):
            p = torch.randint(0,2,(5,))
            z = torch.zeros(3)
            concat = torch.cat((p,z),0).tolist()
            
        # Transform property from list values / Switch only Property_V for while
            nf_csv['Property_V_Velcro'] = concat # Only list can switch values of the column
        

        # Save files
            final_path = os.path.join(self.search_path, save_dir, 'nf_ex' + str(i) + '.csv')
            nf_csv.to_csv(final_path)  
            
    # Getting node features
    def node_feature(self, csv_file, root_dir):
        # Search path
        node_path = os.path.join(self.search_path, root_dir, csv_file)

        # Read csv file to tensor
        nf_csv = pd.read_csv(node_path, index_col=0)
        nf_drop = nf_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        nf = torch.Tensor(nf_drop.values) # dataframe to tensor
        self.x = nf.to(dtype=torch.float32)

        return self.x

    # Data size return
    def __len__(self): 
        return len(self.x)


    # Sampling one specific data from dataset
    def __getitem__(self, index): 
        part_x = torch.FloatTensor(self.x[index])
        return part_x

    # Getting edge_features - edge_index, edge_attribute
    def edge_index(self, csv_file, root_dir):
        # Search path
        edge_index_path = os.path.join(self.search_path, root_dir, 'edge_index', csv_file)

        # Read csv file to tensor
        ef_csv = pd.read_csv(edge_index_path, index_col=0)
        ef = ef_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)


        # edge_index: [2, num_edges], edge_attr: [num_edges, dim_edge_features]
        
        ####################### Recommend to change ################
        ## Edge index

        list_i = []
        list_c = []
       
        for index in range(len(ef.columns)):
            for column in range(len(ef.columns)):
                if ef.iat[index, column] == 1:    # Recommend to change '.iat' to speed up
                    list_i.append(index)
                    list_c.append(column)
                    
        
        tensor_i = torch.tensor(list_i)
        tensor_c = torch.tensor(list_c)
        edge_tensor = torch.cat((tensor_i, tensor_c), dim=0).reshape(2,len(tensor_i))
        edge_index = edge_tensor.to(dtype=torch.int64)

        return edge_index

    def edge_attr(self, csv_file, root_dir):

        edge_attr_path = os.path.join(self.search_path, root_dir, 'edge_attr', csv_file)

        ############################################################
        ## Edge attribute # on, in, attach 임의로 정해서 만들어 놓기
        # edge_attr = torch.Tensor(ef.values)
        ea_csv = pd.read_csv(edge_attr_path, index_col=0)
        ea_drop = ea_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        ea = torch.Tensor(ea_drop.values) # dataframe to tensor
        edge_attr = ea.to(dtype = torch.float32)
        
        return edge_attr


## Print

make_data = MakeDataset(root_path='dataset')


# print(make_data.rand_sample(folder_name='node_features',file_name='nf1.csv',save_dir='node_features', n=13))
x_train = make_data.node_feature(csv_file='nf1.csv', root_dir='node_features')
edge_index_train, edge_attr_train = make_data.edge_index(csv_file='ef0.csv', root_dir='edge_features')

x_test = make_data.node_feature(csv_file='nf1.csv', root_dir='node_features')
edge_index_test, edge_attr_test = make_data.edge_index(csv_file='ef_pick1.csv', root_dir='edge_features')

# print(edge_attr)
# data = x, edge_index, edge_attr


# print("Node Feature:\n",x) #Number of nodes: 8, node feature: 13 (8,13)
# print("\nEdge index:\n",edge_index) #(2,8)
# print("\nEdge attr:\n", edge_attr) #shape [8,8]


## Making graph data
from torch_geometric.data import Data

dataset = Data(x=x_train, edge_index= edge_index_train, edge_attr=edge_attr_train) # Data(x=[8, 13], edge_index=[2, 8], edge_attr=[8, 8])
dataset2 = Data(x= x_test, edge_index= edge_index_test, edge_attr= edge_attr_test)

print("Node Feature:\n",dataset.x) #Number of nodes: 8, node feature: 13 (8,13)
print("\nEdge index:\n",dataset.edge_index) #(2,14)
print("\nEdge attr:\n", dataset.edge_attr) #shape (14,3)


# print(dataset.x) 
# print(dataset.keys) # 'x', 'edge_attr', 'edge_index'
# print(dataset.num_nodes) # 8
# print(dataset.has_isolated_nodes()) #False
# print(dataset.has_self_loops()) #False
# print(dataset.is_directed()) #True


####################################################### GNN #################################################################

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim


# Basically the same as the baseline except we pass edge features 

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"


#### Action input embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#### ActionModel

class ActionModel(nn.Module):
    def __init__(self, hidden_dim, num_action, node_feature_size, edge_feature_size):
        super(ActionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        
        self.convs = [GATConv(in_channels=self.node_feature_size, out_channels=self.hidden_dim, edge_dim=self.edge_feature_size),
                      GATConv(in_channels=self.hidden_dim, out_channels=self.hidden_dim, edge_dim=self.edge_feature_size)]
        
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

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Data format must match! 
        # Type 1) x : float32, edge_index : int64, edge_attr: float32  
        # print(type(x),type(edge_index),type(edge_attr))

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training = self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        
        # print("x",x)
        action_input_emb = x.mean(axis=0)      # x feature를 합치는 과정 / 현재는 mean으로 (추후 변경 예정)
        # print("actopm=input",action_input_emb)
        self.node_features = x
                
        softmax = nn.Softmax(dim=0)
        action_prob = softmax(self.action_layers(action_input_emb))
       
        # action_prob = self.action_layers(action_input_emb)
    
    
        # action_prob = nn.Softmax(self.action_layers(action_input_emb))
        
        
        each_node = []
        for feature in self.node_features:  
            # print(feature) # feature size : [8,8]

            sig = nn.Sigmoid()
            # node_scores.append(nn.Sigmoid(self.node_layers(feature)))
            each_node.append(sig(self.node_layers(feature))) #tensor는 append로 합친 후 concat을 해야 list형식의 tensor형태로 가지고 있는다.
        
            node_scores = torch.cat(each_node, dim=0)
        # print("\n[Each node]",each_node)

        return action_prob, node_scores   
  



#test
hidden_dim = 64
num_action = 3 # [pick, place, pour]
node_feature_size = 13 #노드 feature 크기
edge_feature_size = 6 # 노드 사이의 relation 종류 개수 [on_right,on_left, in_right, in_left, attach, in-grasp]

model = ActionModel(hidden_dim, num_action, node_feature_size, edge_feature_size)
model.to(device)

print("\n[Model]:",model(dataset))

##################################### Calculate Loss / Cross Entropy Loss
input_action_prob, input_node_scores = model(dataset)
# print("\n[Input action probability]:",input_action_prob)
# print("\n[Input node scores]:", input_node_scores)

# CrossEntropyLoss 1) Input (N,C) C= number of classes / Target N where each value is 0
loss = nn.CrossEntropyLoss() 

#Action loss
# target_action_prob = torch.empty(3).random_(2) #세 개 중에 하나만 1로 설정해야함
target_action_prob = torch.tensor([1,0,0], dtype=torch.float32) #세개 중에 하나만 1로 설정해야함
L_action = loss(input_action_prob, target_action_prob)

#Nodescore loss
# target_node_scores = torch.empty(8).random_(2)
target_node_scores = torch.tensor([1,0,0,0,0,0,0,0], dtype=torch.float32) 
L_nodescore = loss(input_node_scores, target_node_scores)

# Total loss
L_total = L_action +L_nodescore


# Print
print("\n[Input_action_prob]:", input_action_prob)
print("\n[Target_action_prob]:", target_action_prob)
print("\n[L_act]:", L_action)
print("\n[Input_node_scores]:",input_node_scores)
print("\n[Target_node_scores]:",target_node_scores)
print("\n[L_nodescore]:",L_nodescore)
print("\nLoss Total:",L_total)

# Load data ## 코드 돌아가는지만 test한 것
train_data = dataset
test_data = dataset2


# Build DataLoader
batch_size = 64
trainloader = DataLoader(
    train_data, batch_size=batch_size
)
testloader = DataLoader(
    test_data, batch_size=batch_size
)


# Set the Training 
model = ActionModel(hidden_dim, num_action, node_feature_size, edge_feature_size).to(device)
data = dataset.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.005, weight_decay=5e-4)

model.train()
for epoch in range(10):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    # CrossEntropyLoss 1) Input (N,C) C= number of classes / Target N where each value is 0
    loss = nn.CrossEntropyLoss() 

    #Action loss
    # target_action_prob = torch.empty(3).random_(2) #세 개 중에 하나만 1로 설정해야함
    target_action_prob = torch.tensor([1,0,0], dtype=torch.float32) #세개 중에 하나만 1로 설정해야함
    L_action = loss(input_action_prob, target_action_prob)

    #Nodescore loss
    # target_node_scores = torch.empty(8).random_(2)
    target_node_scores = torch.tensor([1,0,0,0,0,0,0,0], dtype=torch.float32) 
    L_nodescore = loss(input_node_scores, target_node_scores)

    # Total loss
    L_total = L_action +L_nodescore
    # print(L_total)

    L_action.backward(retain_graph = True) ## Error
    print(L_action)
    optimizer.step()
    
print(input_action_prob, input_node_scores)
print(input_node_scores.max(dim=0)) # Extract indices from max value

# lr = 1e-3
# optimizer = optim.SGD(model.parameters(), lr=lr)
# # Train the Network
# epochs = 10
# for t in range(epochs):
#     print(f'----- Epoch {t+1} -----')
#     print(L_action, L_nodescore)
#     # accuracy, loss = test(testloader, model, loss_fn)







