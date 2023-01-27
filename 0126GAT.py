from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim

# Basically the same as the baseline except we pass edge features 
class GAT(nn.Module):
    def __init__(self, node_feature_size, edge_feature_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.convs = [GATConv(self.node_feature_size, self.hidden_size, self.edge_feature_size),
                      GATConv(self.hidden_size, self.hidden_size, self.edge_feature_size)]
        #self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        #x = self.linear(x)

        return F.relu(x) 

class GCN(nn.Module):
    def __init__(self, node_feature_size, feature_size):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_feature_size, feature_size)
        self.conv2 = GCNConv(feature_size, feature_size)
        self.conv3 = GCNConv(feature_size, feature_size)
    def forward(self, data, attr = 'x'):
        x, edge_index = getattr(data, attr), data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv3(x, edge_index)
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('CUDA:', torch.cuda.is_available(), '     Use << {} >>'.format(device.upper()))
print('PyTorch Version:', torch.__version__)





class ActionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_action, node_feature_size, edge_feature_size):
        super(ActionModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_action = num_action
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        
        self.convs = [GATConv(self.node_feature_size, self.hidden_size, self.edge_feature_size),
                      GATConv(self.hidden_size, self.hidden_size, self.edge_feature_size)]
        
        self.action_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.num_action),
            nn.LeakyReLU(),
        )
        self.node_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.LeakyReLU(),
            nn.Linear(self.input_dim, 1),
        )
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr=edge_attr) # adding edge features here!
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr) # edge features here as well
        
        #action_input_emb = 
        node_features = x
                
        action_prob = nn.Softmax(self.action_layers(action_input_emb))
        
        
        node_scores = []
        for feature in node_features:
            node_scores.append(nn.Sigmoid(self.node_layers(feature)))
        return action_prob, node_scores   
    
    
# Build Train Function
def train(dataloader, model, L_Action, L_NodeScore, optimizer):
    model.train()
    for batch, (x, y) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)
        # Feedforward
        action_prob, node_scores = model(x)
        # Calc. Loss
        L_total = L_Action(action_prob, action_prob_target) + L_NodeScore(node_scores, node_scores_target)
        # Backpropagation
        L_total.backward()
        optimizer.step()
        optimizer.zero_grad()
# Build Test Function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0,0
    with torch.no_grad():
        for x, y in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(x)
            loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    print(f'Test Accuracy: {(100*correct):>0.1f}%     Loss: {loss:>8f} \n')
    return 100*correct, loss
############################################################
# Get the Dataset
############################################################
# Download Dataset
#train_data =
#test_data =
# Build DataLoader
batch_size = 64
trainloader = DataLoader(
    train_data, batch_size=batch_size
)
testloader = DataLoader(
    test_data, batch_size=batch_size
)
############################################################
# Run the Training and Evaluation
############################################################
# Generate the Model
input_dim = 64
hidden_dim = 256
num_action = 3
model = ActionModel(input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_action=num_action).to(device)
# Set the Training Parameters
lr = 1e-3
L_Action = nn.CrossEntropyLoss().to(device)
L_NodeScore = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)
# Train the Network
epochs = 100
for t in range(epochs):
    print(f'----- Epoch {t+1} -----')
    train(trainloader, model, L_Action, L_NodeScore, optimizer)
    #accuracy, loss = test(testloader, model, loss_fn)