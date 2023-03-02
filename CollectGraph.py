import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import  random_split
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import pickle

## Dataset
class ReadDataset():
    def __init__(self, task, sort_edge=True):
        # Search path
        # task: stacking / mixing / ...
        self.search_path = os.path.join(os.getcwd(), 'seq_dataset',task)
        print("search_path:",self.search_path)
        
        self.sort_edge = sort_edge

    # Getting node features
    def node_feature(self, csv_file='nf0.csv'):
        # Search path
        node_path = os.path.join(self.search_path, "node_features", csv_file)

        # Read csv file to tensor
        nf_csv = pd.read_csv(node_path, index_col=0)
        nf = torch.Tensor(nf_csv.values)

        #nf_drop = nf_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        #nf = torch.Tensor(nf_drop.values) # dataframe to tensor
        x = nf.to(dtype=torch.float32)
        return x

    def edge_features(self, order, i):
        # Search path
        edge_attr_path = os.path.join(self.search_path, 'edge_features', order, 'edge_attr', order + '_ea' + str(i) + '.csv')
        
        # Read csv file to tensor
        ea_csv = pd.read_csv(edge_attr_path, index_col=0)
        if self.sort_edge:
            ea_csv = ea_csv.sort_index()

        ea = torch.Tensor(ea_csv.values) # dataframe to tensor
        edge_attr = ea.to(dtype = torch.float32)
        
        # Read edge_index from index(ID) of edge_attr dataframe
        ei_list = []
        for ei in ea_csv.index.to_list():
            _,src,_,_,dest,_ = ei
            ei_list.append(torch.tensor([[int(src)], [int(dest)]]))
        
        edge_index = torch.cat(ei_list, dim=1)

        return edge_index, edge_attr
    
    def graph_data(self, order, i):
        x = self.node_feature()
        edge_index, edge_attr = self.edge_features(order, i)
        
        return Data(x, edge_index, edge_attr)

#data checking
#mix = ReadDataset('mixing_5')
#ex = mix.graph_data(order='1_2_3_5_4', i=3)

#print(ex['x'].shape, ex['edge_index'].shape, ex['edge_attr'].shape)
#print(ex['edge_index'])
#print(ex['edge_attr'])





class CollectGraph():
    def __init__(self, goal_next=False, key_node=False, concat_goal=True, fc_graph=True):
        self.goal_next = goal_next
        self.action_encoder = {'pick':[1, 0, 0], 'place':[0, 1, 0], 'pour':[0, 0, 1]}
        self.key_node = key_node
        self.concat_goal = concat_goal
        self.fc_graph = fc_graph
        #key_node check
        #concat check
            #edge_attr order check
        #fully_connected check? (implement)
        

    def collect_graph(self):
        collect_graph = []

        collect_graph.extend(self.collect_stacking5())
        collect_graph.extend(self.collect_mixing5())

        return collect_graph

    def collect_stacking5(self):
        stacking5_graph = []
        stack_5 = ReadDataset(task='stacking_5')

        x = stack_5.node_feature()

        stack_action = ['pick','place','pick','place','pick','place','pick','place']
        stack_target_obj = [4, 5, 3, 4, 2, 3, 1, 2]

        block_order_list = os.path.join(stack_5.search_path, 'edge_features')
        for order in os.listdir(block_order_list):
            if not(self.goal_next):
                goal_edge_index, goal_edge_attr = stack_5.edge_features(order=order, i=8)

            block_order_num = list(map(int, order.split('_')))

            for i in range(8):
                state_edge_index, state_edge_attr = stack_5.edge_features(order=order, i=i)

                if self.goal_next:
                    goal_edge_index, goal_edge_attr = stack_5.edge_features(order=order, i=i+1)

                cat_edge_attr = torch.cat((state_edge_attr, goal_edge_attr), dim=1)
                
                action_code = torch.Tensor(self.action_encoder[stack_action[i]])

                target_object_index = block_order_num[stack_target_obj[i]-1]
                target_object_score = np.zeros(x.shape[0], dtype=int)
                target_object_score[target_object_index] = 1
                target_object_score = torch.from_numpy(target_object_score).type(torch.FloatTensor)
            
                graph_dict_data = {'input':{},
                                'target':{'action':[],
                                            'object':[]
                                            },
                                'info':{'order':str(),
                                        'step':int()
                                        }
                                            }
                
                graph_dict_data['input']['x'] = x
                graph_dict_data['input']['edge_index'] = state_edge_index
                graph_dict_data['input']['edge_attr'] = cat_edge_attr

                graph_dict_data['target']['action'] = action_code
                graph_dict_data['target']['object'] = target_object_score

                graph_dict_data['info']['order'] = order
                graph_dict_data['info']['step'] = i

                stacking5_graph.append(graph_dict_data)
        return stacking5_graph



    def collect_mixing5(self):
        mixing5_graph = []
        mix_5 = ReadDataset(task='mixing_5')

        x = mix_5.node_feature()
        mix_action = ['pick','place','pick','place','pick','place','pick','place','pick','place','pour']
        mix_target_obj = [5, 6, 4, 6, 3, 6, 2, 6, 1, 6, 7]

        block_order_list = os.path.join(mix_5.search_path, 'edge_features')
        for order in os.listdir(block_order_list):
            if not(self.goal_next):
                goal_edge_index, goal_edge_attr = mix_5.edge_features(order=order, i=8)

            block_order_num = list(map(int, order.split('_')))
            block_order_num.extend([6, 7])

            for i in range(11):
                state_edge_index, state_edge_attr = mix_5.edge_features(order=order, i=i)

                if self.goal_next:
                    goal_edge_index, goal_edge_attr = mix_5.edge_features(order=order, i=i+1)

                cat_edge_attr = torch.cat((state_edge_attr, goal_edge_attr), dim=1)
                
                action_code = torch.Tensor(self.action_encoder[mix_action[i]])
 
                target_object_index = block_order_num[mix_target_obj[i]-1]
                target_object_score = np.zeros(x.shape[0], dtype=int)
                target_object_score[target_object_index] = 1
                target_object_score = torch.from_numpy(target_object_score).type(torch.FloatTensor)
            
                graph_dict_data = {'input':{},
                                'target':{'action':[],
                                            'object':[]
                                            },
                                'info':{'order':str(),
                                        'step':int()
                                        }
                                            }
                
                graph_dict_data['input']['x'] = x
                graph_dict_data['input']['edge_index'] = state_edge_index
                graph_dict_data['input']['edge_attr'] = cat_edge_attr

                graph_dict_data['target']['action'] = action_code
                graph_dict_data['target']['object'] = target_object_score

                graph_dict_data['info']['order'] = order
                graph_dict_data['info']['step'] = i

                mixing5_graph.append(graph_dict_data)
        return mixing5_graph

####Test####
data = CollectGraph()
collected_graph = data.collect_graph()
print("num of data:{}".format(len(collected_graph)))

action_dist = [0, 0, 0]
for data in collected_graph:
    sample = data['target']['action'].tolist()
    if sample == [1, 0, 0]:
        action_dist[0] += 1
    elif sample == [0, 1, 0]:
        action_dist[1] += 1
    else :
        action_dist[2] += 1
print(action_dist)


print("#####dataset split#####")
train, val, test = random_split(collected_graph, [0.8, 0.1, 0.1])
print("num of train:{}".format(len(train)))
print("num of val:{}".format(len(val)))
print("num of test:{}".format(len(test)))

'''
print("#####dataset saved#####")
for i, g in enumerate(train):
    file_path = "./datasets/collected/train/graph_"+str(i)
    with open(file_path, "wb") as outfile:
        pickle.dump(g, outfile)


for i, g in enumerate(val):
    file_path = "./datasets/collected/val/graph_"+str(i)
    with open(file_path, "wb") as outfile:
        pickle.dump(g, outfile)


for i, g in enumerate(test):
    file_path = "./datasets/collected/test/graph_"+str(i)
    with open(file_path, "wb") as outfile:
        pickle.dump(g, outfile)

'''