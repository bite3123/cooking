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

## Dataset
class ReadDataset():
    def __init__(self, task, sort_edge=True, max_node_num = 14, zero_edge_X=False):
        # Search path
        # task: stacking / mixing / ...
        self.search_path = os.path.join(os.getcwd(), 'seq_dataset','tasks', task)
        print("\n[Search_path]\n:",self.search_path)
        
        self.task = task
        self.sort_edge = sort_edge
        self.max_node_num = max_node_num
        self.zero_edge_X = zero_edge_X

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
        #nf = torch.Tensor(nf_csv.values)
        node_id = list(map(str,nf_csv.index.to_list()))
        node_id_to_idx = {}
        for i, n in enumerate(node_id):
            node_id_to_idx[n] = i
        nf = F.pad(torch.Tensor(nf_csv.values), (0,0,0,self.max_node_num-len(node_id_to_idx)), value=0)
        #nf_drop = nf_csv.drop(labels='ID',axis=1) # drop the "ID" column / axis=0 (row), axis=1(column)
        #nf = torch.Tensor(nf_drop.values) # dataframe to tensor
        x = nf.to(dtype=torch.float32)
        return x, node_id_to_idx

    def edge_features(self, order, i):
        # Search path
        edge_attr_path = self.input_csv_file('edge_attr', order, i)
        x, node_id_to_idx = self.node_features(order, i)
        # Read csv file to tensor
        ea_csv = pd.read_csv(edge_attr_path, index_col=0)

        #if self.fully_connected is False:
        #    ea_csv = ea_csv.iloc[:14, :] #drop zero-value edges
        if self.sort_edge:
            ea_csv = ea_csv.sort_index() #sort with ea_id "(src, dest)"
        if self.zero_edge_X:
            ea = torch.Tensor(ea_csv.values) # dataframe to tensor
            for idx, row in ea_csv.iterrows():
                print(idx, row)
                input()
        else:
            ea = torch.Tensor(ea_csv.values) # dataframe to tensor
        edge_attr = ea.to(dtype = torch.float32)
        
        # Read edge_index from index(ID) of edge_attr dataframe
        ei_list = []
        for ei in ea_csv.index.to_list():
            [src, dest] = ei[2:-2].split('\', \'')

            #ei_list.append(torch.tensor([[int(src)], [int(dest)]]))
            ei_list.append(torch.Tensor([[int(node_id_to_idx[src])],[int(node_id_to_idx[dest])]]))

        
        edge_index = torch.cat(ei_list, dim=1)

        return edge_index, edge_attr
    
    def graph_data(self, order, i):
        x, _ = self.node_features(order, i)
        edge_index, edge_attr = self.edge_features(order, i)
        
        return Data(x, edge_index, edge_attr)
#data checking
mix = ReadDataset('stacking_5')
ex = mix.graph_data(order='1_2_3_4_5', i=3)

print(ex['x'].shape, ex['edge_index'].shape, ex['edge_attr'].shape)
print(ex['edge_index'])
print(ex['edge_attr'])

class CollectGraph():
    def __init__(self):
        self.action_encoder = {'pick':[1, 0, 0, 0], 'place':[0, 1, 0, 0], 'pour':[0, 0, 1, 0], 'mix':[0, 0, 0, 1]}        
        self.edge_attr_dim = 7
        
    def collect_graph(self, data_type, auto_save=False, dataset_name=None, OnlyFinalGoal=False, PoseNum=300):
        self.OnlyFinalGoal = OnlyFinalGoal
        self.PoseNum = PoseNum

        collect_graph = []

        collect_graph.extend(self.collect_mixing5(data_type))
        collect_graph.extend(self.collect_stacking5(data_type))
        if auto_save:
            self.dataset_save(data_type, dataset_name, collect_graph)

        return collect_graph
    
    def concat_state_and_goal(self, state_ei, state_ea, goal_ei, goal_ea):
        ei_set = set()
        for i_s in range(state_ei.size(dim=-1)):
            ei_set.add(tuple(state_ei[:,i_s].tolist()))
        for i_g in range(goal_ei.size(dim=-1)):
            ei_set.add(tuple(goal_ei[:,i_g].tolist()))
        cat_ei_list = list(map(lambda x:torch.Tensor(x).unsqueeze(-1), ei_set))
        cat_ei = torch.cat(cat_ei_list, dim=1)
        cat_ei = cat_ei.type(torch.long)

        cat_ei_len = cat_ei.size(dim=-1)

        cat_ea = torch.zeros((cat_ei_len, 2*self.edge_attr_dim))

        for i in range(cat_ei_len):
            for i_s in range(state_ei.size(dim=-1)):
                if torch.equal(cat_ei[:,i],state_ei[:, i_s]):
                    cat_ea[i, :self.edge_attr_dim] = state_ea[i_s,:]
            for i_g in range(goal_ei.size(dim=-1)):
                if torch.equal(cat_ei[:,i],goal_ei[:, i_g]):
                    cat_ea[i, self.edge_attr_dim:] = goal_ea[i_g,:]
        
        return cat_ei, cat_ea
        
    def collect_stacking5(self, data_type):
        stacking5_graph = []
        stack_5 = ReadDataset(task='stacking_5', sort_edge=False)

        stack_action = ['pick','place','pick','place','pick','place','pick','place']
        order_seq_dict = {'1_2_3_4_5':["Box4", "Box5", "Box3", "Box4", "Box2", "Box3", "Box1", "Box2"]}
        if data_type == 'action':
            num_pos = 0
            for demo in os.listdir(os.path.join(stack_5.search_path, '1_2_3_4_5', 'pose')):
            
                for order, stack_target_object in order_seq_dict.items():
                    if self.OnlyFinalGoal:
                        goal_state_num = [8]
                    else:
                        goal_state_num = [8, 6, 4, 2]
                    for i_g in goal_state_num:
                        goal_node_feature, node_id_to_idx = stack_5.node_features(order=order, i=i_g)
                        goal_edge_index, goal_edge_attr = stack_5.edge_features(order=order, i=i_g)

                        for i in range(i_g):
                            x, _ = stack_5.node_features(order=order, i=i)
                            pose_path = stack_5.input_csv_file('pose/'+demo, order, i)
                            #pose_data = torch.Tensor(pd.read_csv(pose_path, index_col=0).values)
                            pose_data = F.pad(torch.Tensor(np.loadtxt(pose_path, delimiter=',')), (0,0,0,stack_5.max_node_num-len(_)), value=0)
                            x = torch.cat([x, pose_data], dim=1)
                            state_edge_index, state_edge_attr = stack_5.edge_features(order=order, i=i)

                            cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                            action_code = torch.Tensor(self.action_encoder[stack_action[i]])

                            target_object_index = node_id_to_idx[stack_target_object[i]]
                            target_object_score = np.zeros(stack_5.max_node_num, dtype=int)
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
                            graph_dict_data['input']['edge_index'] = cat_edge_index
                            graph_dict_data['input']['edge_attr'] = cat_edge_attr

                            graph_dict_data['target']['action'] = action_code
                            graph_dict_data['target']['object'] = target_object_score

                            graph_dict_data['info']['order'] = order
                            graph_dict_data['info']['step'] = i
                            graph_dict_data['info']['demo'] = "stacking_5"
                            graph_dict_data['info']['goal'] = i_g


                            stacking5_graph.append(graph_dict_data)
                if num_pos >=self.PoseNum:
                    break
                num_pos+=1

            return stacking5_graph
        
        elif data_type == 'dynamics':
            for order, stack_target_object in order_seq_dict.items():
                for i in range(8):
                    x, _ = stack_5.node_features(order=order, i=i)
                    state_edge_index, state_edge_attr = stack_5.edge_features(order=order, i=i)

                    goal_node_feature, node_id_to_idx = stack_5.node_features(order=order, i=i+1)
                    goal_edge_index, goal_edge_attr = stack_5.edge_features(order=order, i=i+1)
                    
                    action_code = torch.Tensor(self.action_encoder[stack_action[i]])

                    target_object_index = node_id_to_idx[stack_target_object[i]]
                    target_object_score = np.zeros(stack_5.max_node_num, dtype=int)
                    target_object_score[target_object_index] = 1
                    target_object_score = torch.from_numpy(target_object_score).type(torch.FloatTensor)

                    graph_dict_data = {'state':{},
                                        'goal':{},
                                        'target':{'action':[],
                                                'object':[]
                                                },
                                        'info':{'order':str(),
                                            'step':int(),
                                            'demo':str()
                                            }
                                                }
                    
                    graph_dict_data['state']['x'] = x
                    graph_dict_data['state']['edge_index'] = state_edge_index
                    graph_dict_data['state']['edge_attr'] = state_edge_attr

                    graph_dict_data['goal']['x'] = goal_node_feature
                    graph_dict_data['goal']['edge_index'] = goal_edge_index
                    graph_dict_data['goal']['edge_attr'] = goal_edge_attr

                    graph_dict_data['target']['action'] = action_code
                    graph_dict_data['target']['object'] = target_object_score

                    graph_dict_data['info']['order'] = order
                    graph_dict_data['info']['step'] = i
                    graph_dict_data['info']['demo'] = "stacking_5"

                    stacking5_graph.append(graph_dict_data)
            return stacking5_graph
        
    def collect_mixing5(self, data_type):
        mixing5_graph = []
        mixing_5 = ReadDataset(task='mixing_5', sort_edge=False)

        mixing_action = ['pick','pour','place','pick','pour','place','pick','pour','place','pick','pour','place','pick','pour','place','pick','pour','place']
        order_seq_dict = {'1_2_3_4_5':["Bowl5", "Bowl6", "Table", "Bowl4", "Bowl6", "Table", "Bowl3", "Bowl6", "Table", "Bowl2", "Bowl6", "Table", "Bowl1", "Bowl6", "Table", "Bowl6", "Bowl7","Table"]}
        if data_type == 'action':
            num_pos = 0
            for demo in os.listdir(os.path.join(mixing_5.search_path, '1_2_3_4_5', 'pose')):
            
                for order, mixing_target_object in order_seq_dict.items():

                    if self.OnlyFinalGoal:
                        goal_state_num = [18]
                    else:
                        goal_state_num = [18, 15, 12, 9, 6, 3]
                        
                    for i_g in goal_state_num:
                        goal_node_feature, node_id_to_idx = mixing_5.node_features(order=order, i=i_g)
                        goal_edge_index, goal_edge_attr = mixing_5.edge_features(order=order, i=i_g)

                        for i in range(i_g):
                            x, _ = mixing_5.node_features(order=order, i=i)
                            pose_path = mixing_5.input_csv_file('pose/'+demo, order, i)
                            #pose_data = torch.Tensor(pd.read_csv(pose_path, index_col=0).values)
                            pose_data = F.pad(torch.Tensor(np.loadtxt(pose_path, delimiter=',')), (0,0,0,mixing_5.max_node_num-len(_)), value=0)
                            x = torch.cat([x, pose_data], dim=1)
                            state_edge_index, state_edge_attr = mixing_5.edge_features(order=order, i=i)

                            cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                            action_code = torch.Tensor(self.action_encoder[mixing_action[i]])

                            target_object_index = node_id_to_idx[mixing_target_object[i]]
                            target_object_score = np.zeros(mixing_5.max_node_num, dtype=int)
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
                            graph_dict_data['input']['edge_index'] = cat_edge_index
                            graph_dict_data['input']['edge_attr'] = cat_edge_attr

                            graph_dict_data['target']['action'] = action_code
                            graph_dict_data['target']['object'] = target_object_score

                            graph_dict_data['info']['order'] = order
                            graph_dict_data['info']['step'] = i
                            graph_dict_data['info']['demo'] = "mixing_5"
                            graph_dict_data['info']['goal'] = i_g
                            mixing5_graph.append(graph_dict_data)
                if num_pos >=self.PoseNum:
                    break
                num_pos+=1
            return mixing5_graph
        
        elif data_type == 'dynamics':
            for order, mixing_target_object in order_seq_dict.items():
                for i in range(8):
                    x, _ = mixing_5.node_features(order=order, i=i)
                    state_edge_index, state_edge_attr = mixing_5.edge_features(order=order, i=i)

                    goal_node_feature, node_id_to_idx = mixing_5.node_features(order=order, i=i+1)
                    goal_edge_index, goal_edge_attr = mixing_5.edge_features(order=order, i=i+1)
                    
                    action_code = torch.Tensor(self.action_encoder[mixing_action[i]])

                    target_object_index = node_id_to_idx[mixing_target_object[i]]
                    target_object_score = np.zeros(mixing_5.max_node_num, dtype=int)
                    target_object_score[target_object_index] = 1
                    target_object_score = torch.from_numpy(target_object_score).type(torch.FloatTensor)

                    graph_dict_data = {'state':{},
                                        'goal':{},
                                        'target':{'action':[],
                                                'object':[]
                                                },
                                        'info':{'order':str(),
                                            'step':int(),
                                            'demo':str()
                                            }
                                                }
                    
                    graph_dict_data['state']['x'] = x
                    graph_dict_data['state']['edge_index'] = state_edge_index
                    graph_dict_data['state']['edge_attr'] = state_edge_attr

                    graph_dict_data['goal']['x'] = goal_node_feature
                    graph_dict_data['goal']['edge_index'] = goal_edge_index
                    graph_dict_data['goal']['edge_attr'] = goal_edge_attr

                    graph_dict_data['target']['action'] = action_code
                    graph_dict_data['target']['object'] = target_object_score

                    graph_dict_data['info']['order'] = order
                    graph_dict_data['info']['step'] = i
                    graph_dict_data['info']['demo'] = "mixing_5"

                    mixing5_graph.append(graph_dict_data)
            return mixing5_graph
        
    def data_split(self, total_dataset):
        print("#####dataset split#####")
        proportions = [0.7, 0.15, 0.15]
        lengths = [int(p*len(total_dataset)) for p in proportions]
        lengths[-1] = len(total_dataset) - sum(lengths[:-1])   
        train, val, test = random_split(total_dataset, lengths)
        print("num of train:{}".format(len(train)))
        print("num of val:{}".format(len(val)))
        print("num of test:{}".format(len(test)))
        return train, val, test
    
    def dataset_save(self, data_type, dataset_name, total_dataset):
        train, val, test = self.data_split(total_dataset)
        #data_type: 'action' | 'dynamics'
        dataset_path = "./datasets/" + dataset_name + "/" + data_type
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

stack_mix_5_data = CollectGraph()
action_data = stack_mix_5_data.collect_graph(data_type='action', dataset_name='stacking5_mixing5_Pose_10_OnlyFinalGoal',auto_save=True, OnlyFinalGoal=True, PoseNum=10)
print("num of data:{}".format(len(action_data)))

