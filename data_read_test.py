import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import pickle
## Dataset
class MakeDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path

        # Search path
        search_path = os.path.join(os.getcwd(), self.root_path)
        self.search_path = search_path
        print("search_path:",search_path)

    # Data size return
    def __len__(self): 
        return len(self.x)


    # Sampling one specific data from dataset
    def __getitem__(self, index): 
        part_x = torch.FloatTensor(self.x[index])
        return part_x
            
    # Getting node features
    def node_feature(self, csv_file, root_dir):
        # Search path
        node_path = os.path.join(self.search_path, root_dir, csv_file)

        # Read csv file to tensor
        nf_csv = pd.read_csv(node_path, index_col=0)
        nf = torch.Tensor(nf_csv.values)

        #nf_drop = nf_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        #nf = torch.Tensor(nf_drop.values) # dataframe to tensor
        self.x = nf.to(dtype=torch.float32)

        return self.x

    # Getting edge_features - edge_index, edge_attribute
    def edge_index(self, csv_file, root_dir):
        # Search path
        edge_index_path = os.path.join(self.search_path, root_dir, 'edge_index', csv_file)
    
        # Read csv file to tensor
        ef = pd.read_csv(edge_index_path, index_col=0)
        #print(ef)
        #print(ef.index)
        #print(ef.columns)
        #ef = ef_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)


        # edge_index: [2, num_edges], edge_attr: [num_edges, dim_edge_features]
        
        ####################### Recommend to change ################
        ## Edge index

        list_i = []
        list_c = []
        ID_list = list(map(int, ef.columns))
        for index in range(len(ID_list)):
            for column in range(len(ID_list)):
                if ef.iat[index, column] == 1:    # Recommend to change '.iat' to speed up
                    list_i.append(ID_list[index])
                    list_c.append(ID_list[column])
        '''            
        list_i = []
        list_c = []
        #ID_list = list(map(int, ef.columns))
        for index in range(len(ef.columns)):
            for column in range(len(ef.columns)):
                if ef.iat[index, column] == 1:    # Recommend to change '.iat' to speed up
                    list_i.append(index)
                    list_c.append(column)
        '''

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
        #print(ea_csv.columns)
        #ea_drop = ea_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        ea = torch.Tensor(ea_csv.values) # dataframe to tensor
        ea = ea[:, 0:7]
        edge_attr = ea.to(dtype = torch.float32)
        
        return edge_attr

'''
## Print

make_data = MakeDataset(root_path=os.path.join('seq_dataset', 'stacking_5'))


# print(make_data.rand_sample(folder_name='node_features',file_name='nf1.csv',save_dir='node_features', n=13))
x_train = make_data.node_feature(csv_file='nf0.csv', root_dir='node_features')
edge_index_train= make_data.edge_index(csv_file='ef0.csv', root_dir=os.path.join('ex_1_2_3_4_5', 'edge_features'))
edge_attr_train = make_data.edge_attr(csv_file='ea0.csv', root_dir=os.path.join('ex_1_2_3_4_5', 'edge_features'))
x_test = make_data.node_feature(csv_file='nf0.csv', root_dir='node_features')
edge_index_test = make_data.edge_index(csv_file='ef5.csv', root_dir=os.path.join('ex_1_2_3_4_5', 'edge_features'))
edge_attr_test = make_data.edge_attr(csv_file='ea5.csv', root_dir=os.path.join('ex_1_2_3_4_5', 'edge_features'))
# print(edge_attr)
# data = x, edge_index, edge_attr


# print("Node Feature:\n",x) #Number of nodes: 8, node feature: 13 (8,13)
# print("\nEdge index:\n",edge_index) #(2,8)
# print("\nEdge attr:\n", edge_attr) #shape [8,8]


## Making graph data
dataset = Data(x=x_train, edge_index= edge_index_train, edge_attr=edge_attr_train) # Data(x=[8, 13], edge_index=[2, 8], edge_attr=[8, 8])
dataset2 = Data(x= x_test, edge_index= edge_index_test, edge_attr= edge_attr_test)

print("Node Feature:\n",dataset.x) #Number of nodes: 8, node feature: 13 (8,13)
print("\nEdge index:\n",dataset.edge_index) #(2,14)
print("\nEdge attr:\n", dataset.edge_attr) #shape (14,3)

print("Node Feature:\n",dataset2.x) #Number of nodes: 8, node feature: 13 (8,13)
print("\nEdge index:\n",dataset2.edge_index) #(2,14)
print("\nEdge attr:\n", dataset2.edge_attr) #shape (14,3)
'''
def to_fully_connected(state_edge_index, state_edge_attr):
    edge_index_template = np.ones((9, 9), dtype=int)
    for idx in range(9):
        edge_index_template[idx][idx] = 0
    #print(state_edge_index.size(1))
    for idx in range(state_edge_index.size(1)):
        src, dest = (state_edge_index[0][idx].item(), state_edge_index[1][idx].item())
        edge_index_template[src][dest] = 0
        #edge_index_template[src][dest] = 1

    for src in range(9):
        for dest in range(9):
            if edge_index_template[src][dest] == 1:
                state_edge_index = torch.cat((state_edge_index, torch.tensor([[src],[dest]])), dim=1)
                #state_edge_index[0].append(src)
                #state_edge_index[1].append(dest)
                state_edge_attr = torch.cat((state_edge_attr, torch.zeros(1, 7)), dim=0)
                #state_edge_attr.append(np.zeros((13), dtype=int))
    #print(state_edge_index.shape)
    #print(state_edge_attr.shape)
    #input()
    #print(block_order, edge_index_template)
    return state_edge_index, state_edge_attr

def concat_state_n_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr):
    """
        state_edge_index: 2 X 72
        state_edge_attr: 72 X 13
        
        goal_edge_index: 2 X 14?
        goal_edge_attr: 14? X 13

        =>
        cat_edge_index: 2 X 72
        cat_edge_attr: 72 X 26
    """

    cat_list = []
    for idx in range(len(state_edge_index[0])):
        state_index = state_edge_index[:,idx]
        matched = False
        for g_idx in range(len(goal_edge_index[0])):
            goal_index = goal_edge_index[:, g_idx]
            #print(state_index)
            #print(goal_index)
            #input()
            if torch.equal(state_index, goal_index):
                matched = True
                #print(state_edge_attr[idx].shape)
                #print(goal_edge_attr[g_idx].shape)

                #print(torch.cat((state_edge_attr[idx], goal_edge_attr[g_idx]), dim=0))
                #input()

                #state_edge_attr[idx] = torch.cat((state_edge_attr[idx], goal_edge_attr[g_idx]), dim=0)
                cat_list.append(torch.cat((state_edge_attr[idx], goal_edge_attr[g_idx]), dim=0))
                break
        if matched == False:
            cat_list.append(torch.cat((state_edge_attr[idx], torch.zeros(7)), dim=0))
            
    cat_edge_attr = torch.stack(cat_list, dim=0)
    #print(state_edge_index.shape)
    #print(cat_edge_attr.shape)

    #input()

    return state_edge_index, cat_edge_attr

def key_node_list(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr):
    state_index_set = set()
    goal_index_set = set()
    state_edge_index = torch.t(state_edge_index)
    goal_edge_index = torch.t(goal_edge_index)
    #print(state_edge_index)
    #print(goal_edge_index)
    for idx in state_edge_index:
        temp = tuple(map(int, idx.tolist()))
        state_index_set.add(temp)
    for idx in goal_edge_index:
        temp = tuple(map(int, idx.tolist()))
        goal_index_set.add(temp)
    key_index_set = (state_index_set|goal_index_set) - (state_index_set&goal_index_set)
    #print(key_index_set)
    key_index_list = []
    for idx in key_index_set:
        key_index_list.append(torch.tensor(list(idx)))
    key_edge_index = torch.stack(key_index_list, dim=1)
    key_edge_index = torch.t(key_edge_index)
    #print(key_edge_index)

    key_node_set = set()
    for i in torch.flatten(key_edge_index).tolist():
        key_node_set.add(i)
    key_node_list = torch.tensor(list(key_node_set))
    #print(key_node_list)
    
    #key_edge_index, key_edge_attr = subgraph(subset=key_node_list, edge_index=torch.t(state_edge_index), edge_attr=state_edge_attr)

    #input()



    return key_node_list



def stacking_5_dataset():
    stacking_dataset = []

    make_data = MakeDataset(root_path=os.path.join('seq_dataset', 'stacking_5'))
    x = make_data.node_feature(csv_file='nf0.csv', root_dir='node_features')

    action_sequence = ['pick','place','pick','place','pick','place','pick','place']
    action_encoder = {'pick':[1, 0, 0], 'place':[0, 1, 0], 'pour':[0, 0, 1]}
    target_object_sequence = [4, 5, 3, 4, 2, 3, 1, 2]

    block_order_list = os.path.join(make_data.search_path, 'edge_data')
    for block_order in os.listdir(block_order_list):
        #goal_edge_index= make_data.edge_index(csv_file='ef8.csv', root_dir=os.path.join('edge_data', block_order, 'edge_features'))
        #goal_edge_attr = make_data.edge_attr(csv_file='ea8.csv', root_dir=os.path.join('edge_data', block_order, 'edge_features'))
        
        #goal_edge_index, goal_edge_attr = to_fully_connected(goal_edge_index, goal_edge_attr)

        block_order_num = list(map(int, block_order.split('_')))
        #print(block_order_num)
        for i in range(8):
            state_edge_index= make_data.edge_index(csv_file='ef'+str(i)+'.csv', root_dir=os.path.join('edge_data', block_order, 'edge_features'))
            state_edge_attr = make_data.edge_attr(csv_file='ea'+str(i)+'.csv', root_dir=os.path.join('edge_data', block_order, 'edge_features'))
            
            goal_edge_index= make_data.edge_index(csv_file='ef'+str(i+1)+'.csv', root_dir=os.path.join('edge_data', block_order, 'edge_features'))
            goal_edge_attr = make_data.edge_attr(csv_file='ea'+str(i+1)+'.csv', root_dir=os.path.join('edge_data', block_order, 'edge_features'))
            
            key_node = key_node_list(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)

            #state_edge_index, state_edge_attr = to_fully_connected(state_edge_index, state_edge_attr)
            #goal_edge_index, goal_edge_attr = to_fully_connected(goal_edge_index, goal_edge_attr) 

            cat_edge_index, cat_edge_attr = concat_state_n_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)

            cat_edge_index, cat_edge_attr = subgraph(subset=key_node, edge_index=cat_edge_index, edge_attr=cat_edge_attr)
            #print(cat_edge_index)
            #print(cat_edge_attr)
            #input()

            action_code = torch.Tensor(action_encoder[action_sequence[i]])

            target_object_index = block_order_num[target_object_sequence[i]-1]
            target_object_score = np.zeros(x.shape[0], dtype=int)
            target_object_score[target_object_index] = 1
            target_object_score = torch.from_numpy(target_object_score).type(torch.FloatTensor)

            #write data to dictionary
            graph_dict_data = {'input':{'state':{},
                                        'goal':{},
                                        'key':{}
                                        },
                            'target':{'action':[],
                                        'object':[]
                                        },
                            'info':{'order':str(),
                                    'step':int()
                                    }
                                        }
            
            graph_dict_data['input']['state']['x'] = x
            graph_dict_data['input']['state']['edge_index'] = state_edge_index
            graph_dict_data['input']['state']['edge_attr'] = state_edge_attr

            graph_dict_data['input']['goal']['x'] = x
            graph_dict_data['input']['goal']['edge_index'] = goal_edge_index
            graph_dict_data['input']['goal']['edge_attr'] = goal_edge_attr

            graph_dict_data['input']['key']['edge_index'] = cat_edge_index
            graph_dict_data['input']['key']['edge_attr'] = cat_edge_attr
            graph_dict_data['input']['key']['key_node'] = key_node

            graph_dict_data['target']['action'] = action_code
            graph_dict_data['target']['object'] = target_object_score

            graph_dict_data['info']['order'] = block_order
            graph_dict_data['info']['step'] = i
            #print(graph_dict_data['input']['state']['edge_index'])
            #print(graph_dict_data['input']['state']['edge_attr'].shape)
            #input()

            stacking_dataset.append(graph_dict_data)

    return stacking_dataset


stacking_dataset = stacking_5_dataset()
print("the num of data:", len(stacking_dataset))#120X8=960

for g in stacking_dataset:
    #print(g)
    print("#############Checking#############")

    print("Stacking Order: ", g['info']['order'])
    print("Step in order: ", g['info']['step'])
    print("\nCurrent State:")
    print("\tedge_index:\n",g['input']['state']['edge_index'].numpy())
    print("\tedge_attr:\n",g['input']['state']['edge_attr'].numpy())
    input()

#print(stacking_dataset[0])
#print(stacking_dataset[8]['input']['state']['edge_index'])
#print(stacking_dataset[8]['input']['goal']['edge_index'])
#print(stacking_dataset[8]['info'])\
#save each graph in json file
'''
for i, g in enumerate(stacking_dataset):
    file_path = "./stacking_dataset_nopos_nofc/stacking_"+str(i)
    with open(file_path, "wb") as outfile:
        pickle.dump(g, outfile)
'''


'''
with open("./stacking_dataset/stacking_"+str(10), "rb") as file:
    load_data = pickle.load(file)
    print(load_data)
'''