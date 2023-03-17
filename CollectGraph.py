import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import  random_split
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import torch.nn.functional as F
import pickle
import natsort


# [File path] /home/jeni/Desktop/dataloader/seq_dataset

# [Search_path] /home/jeni/Desktop/dataloader/seq_dataset/tasks/stacking_v4/1234_5

# [Example] 1234_5

# [Problem] stacking_v4




## Dataset
class ReadDataset():
    def __init__(self, task, sort_edge=True, without_pos=False, fully_connected=True):
        # Search path
        # task: stacking / mixing / ...
        self.search_path = os.path.join(os.getcwd(), 'seq_dataset','tasks', task)
        print("\n[Search_path]\n:",self.search_path)
        
        self.task = task
        self.sort_edge = sort_edge
        self.without_pos = without_pos
        self.fully_connected = fully_connected


    def input_csv_file(self, feature, order, i):    
        feature_path = os.path.join(self.search_path, order, feature)
        file_list = natsort.natsorted(os.listdir(feature_path))
        input_path = os.path.join(feature_path, file_list[i])
        
        return input_path

   
    # Getting node features
    def node_features(self,  order, i):
        # Search path
        node_path = self.input_csv_file('node_features', order, i)

        # Read csv file to tensor
        nf_csv = pd.read_csv(node_path, index_col=0)
        node_id = list(map(str,nf_csv.index.to_list()))
        node_id_to_idx = {}
        for i, n in enumerate(node_id):
            node_id_to_idx[n] = i
        nf = F.pad(torch.Tensor(nf_csv.values), (0,0,0,13-len(node_id_to_idx)), value=0)

        #nf_drop = nf_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        #nf = torch.Tensor(nf_drop.values) # dataframe to tensor
        x = nf.to(dtype=torch.float32)
        return x

    def edge_index(self, order, i):
        # Search path
        edge_inx_path = self.input_csv_file('edge_index', order, i)
        
        # Read csv file to tensor
        ei_csv = pd.read_csv(edge_inx_path, index_col=0)
        ei = torch.Tensor(ei_csv.values) # dataframe to tensor
        edge_index = ei.to(dtype = torch.float32)

        return edge_index


    def edge_attr(self, order, i):
        # Search path
        edge_attr_path = self.input_csv_file('edge_attr', order, i)
        
        # Read csv file to tensor
        ea_csv = pd.read_csv(edge_attr_path, index_col=0)
        pos = ea_csv.iloc[:,7:]

        if self.fully_connected is False:
            ea_csv = ea_csv.iloc[:14, :] #drop zero-value edges
            pos = pos.iloc[:14, :]
        if self.without_pos:
            ea_csv = ea_csv.iloc[:, :7]
        if self.sort_edge:
            ea_csv = ea_csv.sort_index()
            pos = pos.sort_index()

        ea = torch.Tensor(ea_csv.values) # dataframe to tensor
        edge_attr = ea.to(dtype = torch.float32)
        
        # Read edge_index from index(ID) of edge_attr dataframe
        # ei_list = []
        # for ei in ea_csv.index.to_list():
        #     _,src,_,_,dest,_ = ei
        #     ei_list.append(torch.tensor([[int(src)], [int(dest)]]))
        
        # edge_index = torch.cat(ei_list, dim=1)

        return edge_attr #, pos
    
    def graph_data(self, order, i):
        x = self.node_features(order, i)
        edge_index = self.edge_index(order, i)
        edge_attr = self.edge_attr(order, i)
        
        return Data(x, edge_index, edge_attr)

#data checking
stack = ReadDataset('stacking_5')

ex = stack.graph_data(order='1_2_3_4_5', i=3)

print(ex['x'].shape, ex['edge_index'].shape, ex['edge_attr'].shape)
print(ex['x'])
print(ex['edge_index'])
print(ex['edge_attr'])
print("end")


class ReadDataset_VariNodeNum():
    def __init__(self, task, sort_edge=False, without_pos=False, max_node_num=9):
        # Search path
        # task: stacking / mixing / ...
        self.search_path = os.path.join(os.getcwd(), 'seq_dataset',task)
        print("search_path:",self.search_path)
        
        self.sort_edge = sort_edge
        self.without_pos = without_pos
        self.max_node_num = max_node_num

    # Getting node features
    def node_feature(self, order, i):
        # Search path
        node_feature_path = os.path.join(self.search_path, 'node_features', order, order + '_nf' + str(i) + '.csv')

        # Read csv file to tensor
        nf_csv = pd.read_csv(node_feature_path, index_col=0)
        node_id = list(map(str,nf_csv.index.to_list()))
        node_id_to_idx = {}
        node_idx_to_id = {}
        for i, n in enumerate(node_id):
            node_id_to_idx[n] = i
            node_idx_to_id[i] = n
        nf = F.pad(torch.Tensor(nf_csv.values), (0,0,0,self.max_node_num-len(node_id_to_idx)), value=0)

        #nf_drop = nf_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        #nf = torch.Tensor(nf_drop.values) # dataframe to tensor
        x = nf.to(dtype=torch.float32)
        return x, node_id_to_idx, node_idx_to_id

    def edge_features(self, order, i):
        # Search path
        edge_attr_path = os.path.join(self.search_path, 'edge_features', order, 'edge_attr', order + '_ea' + str(i) + '.csv')
        
        # Read csv file to tensor
        ea_csv = pd.read_csv(edge_attr_path, index_col=0)
        ea_csv = ea_csv.iloc[:14, :] #drop zero-value edges
        if self.without_pos:
            ea_csv = ea_csv.iloc[:, :7]
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


class CollectGraph_VariNodeNum():
    def __init__(self, goal_next=False, key_node=True, fc_graph=False):

        self.goal_next = goal_next
        self.action_encoder = {'pick':[1, 0, 0], 'place':[0, 1, 0], 'pour':[0, 0, 1]}
        self.key_node = key_node
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
        stack_5 = ReadDataset_VariNodeNum(task='stacking_5')

        stack_action = ['pick','place','pick','place','pick','place','pick','place']
        stack_target_obj = [4, 5, 3, 4, 2, 3, 1, 2]

        block_order_list = os.path.join(stack_5.search_path, 'edge_features')
        for order in os.listdir(block_order_list):
            if not(self.goal_next):
                goal_edge_index, goal_edge_attr = stack_5.edge_features(order=order, i=8)

            block_order_num = list(map(int, order.split('_')))

            for i in range(8):
                x, node_id_to_idx = stack_5.node_feature(order=order, i=i)

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
                                        'step':int(),
                                        'demo':str()
                                        }
                                            }
                
                graph_dict_data['input']['x'] = x
                graph_dict_data['input']['edge_index'] = state_edge_index
                graph_dict_data['input']['edge_attr'] = cat_edge_attr

                graph_dict_data['target']['action'] = action_code
                graph_dict_data['target']['object'] = target_object_score

                graph_dict_data['info']['order'] = order
                graph_dict_data['info']['step'] = i
                graph_dict_data['info']['demo'] = "stacking_5"

                stacking5_graph.append(graph_dict_data)


        return stacking5_graph



    def collect_mixing5(self):
        mixing5_graph = []
        mix_5 = ReadDataset(task='mixing_5')

        x = mix_5.node_feature()
        mix_action = ['pick','place','pick','place','pick','place','pick','place','pick','place','pick','pour','place']
        mix_target_obj = [5, 6, 4, 6, 3, 6, 2, 6, 1, 6, 6, 7, 8]

        block_order_list = os.path.join(mix_5.search_path, 'edge_features')
        for order in os.listdir(block_order_list):
            if not(self.goal_next):
                goal_edge_index, goal_edge_attr = mix_5.edge_features(order=order, i=13)

            block_order_num = list(map(int, order.split('_')))
            block_order_num.extend([6, 7, 8])

            for i in range(13):
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
                                        'step':int(),
                                        'demo':str()
                                        }
                                            }
                
                graph_dict_data['input']['x'] = x
                graph_dict_data['input']['edge_index'] = state_edge_index
                graph_dict_data['input']['edge_attr'] = cat_edge_attr

                graph_dict_data['target']['action'] = action_code
                graph_dict_data['target']['object'] = target_object_score

                graph_dict_data['info']['order'] = order
                graph_dict_data['info']['step'] = i
                graph_dict_data['info']['demo'] = "mixing_5"

                mixing5_graph.append(graph_dict_data)

        return mixing5_graph

class CollectGraph():
    def __init__(self, goal_next=False, key_node=False, fc_graph=True):
        self.goal_next = goal_next
        self.action_encoder = {'pick':[1, 0, 0], 'place':[0, 1, 0], 'pour':[0, 0, 1]}
        self.key_node = key_node
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
        stack_5 = ReadDataset(task='stacking_5', sort_edge=False,fully_connected=False)

        x = stack_5.node_feature()

        stack_action = ['pick','place','pick','place','pick','place','pick','place']
        stack_target_obj = [4, 5, 3, 4, 2, 3, 1, 2]

        block_order_list = os.path.join(stack_5.search_path, 'edge_features')
        for order in os.listdir(block_order_list):
            if not(self.goal_next):
                goal_edge_index, goal_edge_attr, goal_pos = stack_5.edge_features(order=order, i=8)

            block_order_num = list(map(int, order.split('_')))

            for i in range(8):
                state_edge_index, state_edge_attr, state_pos = stack_5.edge_features(order=order, i=i)

                if self.goal_next:
                    goal_edge_index, goal_edge_attr, goal_pos = stack_5.edge_features(order=order, i=i+1)

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
                                        'step':int(),
                                        'demo':str()
                                        }
                                            }
                
                graph_dict_data['input']['x'] = x
                graph_dict_data['input']['edge_index'] = state_edge_index
                graph_dict_data['input']['edge_attr'] = cat_edge_attr

                graph_dict_data['target']['action'] = action_code
                graph_dict_data['target']['object'] = target_object_score

                graph_dict_data['info']['order'] = order
                graph_dict_data['info']['step'] = i
                graph_dict_data['info']['demo'] = "stacking_5"

                stacking5_graph.append(graph_dict_data)
        return stacking5_graph



    def collect_mixing5(self):
        mixing5_graph = []
        mix_5 = ReadDataset(task='mixing_5', fully_connected=False)

        x = mix_5.node_feature()
        mix_action = ['pick','place','pick','place','pick','place','pick','place','pick','place','pick','pour','place']
        mix_target_obj = [5, 6, 4, 6, 3, 6, 2, 6, 1, 6, 6, 7, 8]

        block_order_list = os.path.join(mix_5.search_path, 'edge_features')
        for order in os.listdir(block_order_list):
            if not(self.goal_next):
                goal_edge_index, goal_edge_attr = mix_5.edge_features(order=order, i=13)

            block_order_num = list(map(int, order.split('_')))
            block_order_num.extend([6, 7, 8])

            for i in range(13):
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
                                        'step':int(),
                                        'demo':str()
                                        }
                                            }
                
                graph_dict_data['input']['x'] = x
                graph_dict_data['input']['edge_index'] = state_edge_index
                graph_dict_data['input']['edge_attr'] = cat_edge_attr

                graph_dict_data['target']['action'] = action_code
                graph_dict_data['target']['object'] = target_object_score

                graph_dict_data['info']['order'] = order
                graph_dict_data['info']['step'] = i
                graph_dict_data['info']['demo'] = "mixing_5"

                mixing5_graph.append(graph_dict_data)
        return mixing5_graph

####Test####
# data = CollectGraph()
# collected_graph = data.collect_graph()
# print("num of data:{}".format(len(collected_graph)))

# action_dist = [0, 0, 0]
# for data in collected_graph:
#     sample = data['target']['action'].tolist()
#     if sample == [1, 0, 0]:
#         action_dist[0] += 1
#     elif sample == [0, 1, 0]:
#         action_dist[1] += 1
#     else :
#         action_dist[2] += 1
# print(action_dist)


# print("#####dataset split#####")
# proportions = [0.8, 0.1, 0.1]
# lengths = [int(p*len(collected_graph)) for p in proportions]
# lengths[-1] = len(collected_graph) - sum(lengths[:-1])   
# train, val, test = random_split(collected_graph, lengths)
# print("num of train:{}".format(len(train)))
# print("num of val:{}".format(len(val)))
# print("num of test:{}".format(len(test)))


# for g in collected_graph:
#     #print(g)
#     print("#############Checking#############")

#     print("Stacking Order: ", g['info']['order'])
#     print("Step in order: ", g['info']['step'])
#     print("\nCurrent State:")
#     print("\tedge_index:\n",g['input']['edge_index'].numpy())
#     print("\tedge_attr:\n",g['input']['edge_attr'].numpy())

#     print("target action:\n",g['target']['action'])
#     print("target object:\n", g['target']['object'])
#     input()
'''

dataset_path = "./datasets/stack_mix_fc_test"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
if not os.path.exists(dataset_path+"/train"):
    os.makedirs(dataset_path+"/train")
if not os.path.exists(dataset_path+"/val"):
    os.makedirs(dataset_path+"/val")
if not os.path.exists(dataset_path+"/test"):
    os.makedirs(dataset_path+"/test")

print("#####dataset saved#####")
for i, g in enumerate(train):
    file_path = os.path.join(dataset_path,"train","graph_"+str(i))
    with open(file_path, "wb") as outfile:
        pickle.dump(g, outfile)


for i, g in enumerate(val):
    file_path = os.path.join(dataset_path,"val","graph_"+str(i))
    with open(file_path, "wb") as outfile:
        pickle.dump(g, outfile)


for i, g in enumerate(test):
    file_path = os.path.join(dataset_path,"test","graph_"+str(i))
    with open(file_path, "wb") as outfile:
        pickle.dump(g, outfile)
'''