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
    def __init__(self, task, sort_edge=True, max_node_num = 33, zero_edge_X=False):
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

        x = torch.Tensor(nf_csv.values).to(dtype=torch.float32)
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
    
class CollectGraph():
    def __init__(self):
        self.action_encoder = {'pick':[1, 0, 0], 'place':[0, 1, 0], 'pour':[0, 0, 1]}        
        self.edge_attr_dim = 7
        
    def collect_graph(self, data_type, auto_save=False, dataset_name=None, OnlyFinalGoal=False):
        self.OnlyFinalGoal = OnlyFinalGoal

        collect_graph = []

        collect_graph.extend(self.collect_stacking5(data_type))
        collect_graph.extend(self.collect_stacking_init2(data_type))
        collect_graph.extend(self.collect_stacking_init3(data_type))
        collect_graph.extend(self.collect_stacking_init3_replace(data_type))
        collect_graph.extend(self.collect_stacking_init3_reverse(data_type))
        collect_graph.extend(self.collect_mixing_5(data_type))
        collect_graph.extend(self.collect_mixing_4(data_type))
        collect_graph.extend(self.collect_mixing_3(data_type))
        collect_graph.extend(self.collect_mixing_2(data_type))
        collect_graph.extend(self.collect_mixing_withbox5(data_type)) 
        collect_graph.extend(self.collect_mixing_withbox4(data_type)) 
        collect_graph.extend(self.collect_mixing_withbox3(data_type)) 
        collect_graph.extend(self.collect_mixing_withbox2(data_type)) 
        collect_graph.extend(self.collect_cleaning_box(data_type)) 
        collect_graph.extend(self.collect_cleaning_init2(data_type))
        collect_graph.extend(self.collect_cleaning_init3(data_type))
        collect_graph.extend(self.collect_cleaning_init4(data_type)) 
        collect_graph.extend(self.collect_cleaning_init5(data_type))
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
        #order_seq_dict = {'1_2_3_4_5':["Box4", "Box5", "Box3", "Box4", "Box2", "Box3", "Box1", "Box2"]}
        stack_target_object = ["Box4", "Box5", "Box3", "Box4", "Box2", "Box3", "Box1", "Box2"]
        if data_type == 'action':
            #for order, stack_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [8]
                else:
                    goal_state_num = [8, 6, 4, 2]
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = stack_5.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = stack_5.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = stack_5.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = stack_5.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[stack_action[i]])

                        target_object_index = node_id_to_idx[stack_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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

            return stacking5_graph
                            
    def collect_stacking_init2(self, data_type):
        stacking_init2_graph = []
        stacking_init2 = ReadDataset(task='stacking_init2', sort_edge=False)

        stack_action = ['pick','place','pick','place']

        stack_target_object = ["Box2", "Box3", "Box1", "Box2"]
        if data_type == 'action':
            #for order, stack_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [4]
                else:
                    goal_state_num = [4, 2]
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = stacking_init2.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = stacking_init2.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = stacking_init2.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = stacking_init2.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[stack_action[i]])

                        target_object_index = node_id_to_idx[stack_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "stacking_init2"
                        graph_dict_data['info']['goal'] = i_g

                        stacking_init2_graph.append(graph_dict_data)
            return stacking_init2_graph
        
    def collect_stacking_init3(self, data_type):
        stacking_init3_graph = []
        stacking_init3 = ReadDataset(task='stacking_init3', sort_edge=False)

        stack_action = ['pick','place']

        stack_target_object = ["Box1", "Box2"]
        if data_type == 'action':
            #for order, stack_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [2]
                else:
                    goal_state_num = [2]
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = stacking_init3.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = stacking_init3.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = stacking_init3.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = stacking_init3.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[stack_action[i]])

                        target_object_index = node_id_to_idx[stack_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "stacking_init3"
                        graph_dict_data['info']['goal'] = i_g

                        stacking_init3_graph.append(graph_dict_data)
            return stacking_init3_graph

    def collect_stacking_init3_replace(self, data_type):
        stacking_init3_replace_graph = []
        stacking_init3_replace = ReadDataset(task='stacking_init3_replace', sort_edge=False)

        stack_action = ['pick','place','pick','place','pick','place','pick','place']

        stack_target_object = ["Box3", "Region_Free","Box4", "Region_Free", "Box2", "Box5", "Box3", "Box2"]
        if data_type == 'action':
            #for order, stack_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [8]
                else:
                    goal_state_num = [8,6,4,2]
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = stacking_init3_replace.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = stacking_init3_replace.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = stacking_init3_replace.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = stacking_init3_replace.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[stack_action[i]])

                        target_object_index = node_id_to_idx[stack_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "stacking_init3_replace"
                        graph_dict_data['info']['goal'] = i_g

                        stacking_init3_replace_graph.append(graph_dict_data)
            return stacking_init3_replace_graph      
        
    def collect_stacking_init3_reverse(self, data_type):
        stacking_init3_reverse_graph = []
        stacking_init3_reverse = ReadDataset(task='stacking_init3_reverse', sort_edge=False)

        stack_action = ['pick','place','pick','place','pick','place','pick','place','pick','place']

        stack_target_object = ["Box3", "Region_Free","Box4", "Region_Free", "Box5","Region_Free", "Box4", "Box3", "Box5", "Box4"]
        if data_type == 'action':
            #for order, stack_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [10]
                else:
                    goal_state_num = [10,8,6,4,2]
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = stacking_init3_reverse.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = stacking_init3_reverse.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = stacking_init3_reverse.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = stacking_init3_reverse.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[stack_action[i]])

                        target_object_index = node_id_to_idx[stack_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "stacking_init3_reverse"
                        graph_dict_data['info']['goal'] = i_g

                        stacking_init3_reverse_graph.append(graph_dict_data)
            return stacking_init3_reverse_graph      


                  
    def collect_mixing_5(self, data_type):
        mixing_5_graph = []
        mixing_5 = ReadDataset(task='mixing_5', sort_edge=False)

        mixing_action = ['pick','place','pick','place',
                         'pick','pour','place','pick','pour','place','pick','pour','place',
                         'pick','pour','place','pick','pour','place','pick','pour','place']
        mixing_target_object = ["Bowl6","Region_Pour","Bowl7","Region_Pour",
                                "Bowl1", "Bowl6", "Region_Bw1", "Bowl2", "Bowl6", "Region_Bw2", "Bowl3", "Bowl6", "Region_Bw3",
                                "Bowl4", "Bowl6", "Region_Bw4", "Bowl5", "Bowl6", "Region_Bw5", "Bowl6", "Bowl7","Region_Bw6"]
        if data_type == 'action':
        
            #for order, mixing_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [22]
                else:
                    goal_state_num = [22, 19, 16, 13, 10, 7]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = mixing_5.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = mixing_5.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = mixing_5.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = mixing_5.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[mixing_action[i]])

                        target_object_index = node_id_to_idx[mixing_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        mixing_5_graph.append(graph_dict_data)
            return mixing_5_graph

    def collect_mixing_4(self, data_type):
        mixing_4_graph = []
        mixing_4 = ReadDataset(task='mixing_4', sort_edge=False)

        mixing_action = ['pick','place','pick','place',
                         'pick','pour','place','pick','pour','place','pick','pour','place',
                         'pick','pour','place','pick','pour','place']
        mixing_target_object = ["Bowl6","Region_Pour","Bowl7","Region_Pour",
                                "Bowl1", "Bowl6", "Region_Bw1", "Bowl2", "Bowl6", "Region_Bw2", "Bowl3", "Bowl6", "Region_Bw3",
                                "Bowl4", "Bowl6", "Region_Bw4", "Bowl6", "Bowl7","Region_Bw6"]
        if data_type == 'action':
        
            #for order, mixing_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [19]
                else:
                    goal_state_num = [19, 16, 13, 10, 7]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = mixing_4.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = mixing_4.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = mixing_4.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = mixing_4.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[mixing_action[i]])

                        target_object_index = node_id_to_idx[mixing_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "mixing_4"
                        graph_dict_data['info']['goal'] = i_g
                        mixing_4_graph.append(graph_dict_data)
            return mixing_4_graph
    def collect_mixing_3(self, data_type):
        mixing_3_graph = []
        mixing_3 = ReadDataset(task='mixing_3', sort_edge=False)

        mixing_action = ['pick','place','pick','place',
                         'pick','pour','place','pick','pour','place','pick','pour','place',
                         'pick','pour','place']
        mixing_target_object = ["Bowl6","Region_Pour","Bowl7","Region_Pour",
                                "Bowl1", "Bowl6", "Region_Bw1", "Bowl2", "Bowl6", "Region_Bw2", "Bowl3", "Bowl6", "Region_Bw3",
                                "Bowl6", "Bowl7","Region_Bw6"]
        if data_type == 'action':
        
            #for order, mixing_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [16]
                else:
                    goal_state_num = [16, 13, 10, 7]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = mixing_3.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = mixing_3.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = mixing_3.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = mixing_3.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[mixing_action[i]])

                        target_object_index = node_id_to_idx[mixing_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "mixing_3"
                        graph_dict_data['info']['goal'] = i_g
                        mixing_3_graph.append(graph_dict_data)
            return mixing_3_graph
    def collect_mixing_2(self, data_type):
        mixing_2_graph = []
        mixing_2 = ReadDataset(task='mixing_2', sort_edge=False)

        mixing_action = ['pick','place','pick','place',
                         'pick','pour','place','pick','pour','place','pick','pour','place']
        mixing_target_object = ["Bowl6","Region_Pour","Bowl7","Region_Pour",
                                "Bowl1", "Bowl6", "Region_Bw1", "Bowl2", "Bowl6", "Region_Bw2","Bowl6", "Bowl7","Region_Bw6"]
        if data_type == 'action':
        
            #for order, mixing_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [13]
                else:
                    goal_state_num = [13, 10, 7]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = mixing_2.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = mixing_2.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = mixing_2.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = mixing_2.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[mixing_action[i]])

                        target_object_index = node_id_to_idx[mixing_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "mixing_2"
                        graph_dict_data['info']['goal'] = i_g
                        mixing_2_graph.append(graph_dict_data)
            return mixing_2_graph
        

                  
    def collect_mixing_withbox5(self, data_type):
        mixing_withbox5_graph = []
        mixing_withbox5 = ReadDataset(task='mixing_withbox5', sort_edge=False)

        mixing_action = ['pick','place','pick','place',
                         'pick','place','pick','place','pick','place','pick','place','pick','place',
                         'pick','pour','place','pick','pour','place','pick','pour','place',
                         'pick','pour','place','pick','pour','place','pick','pour','place']
        mixing_target_object = ["Bowl6","Region_Pour","Bowl7","Region_Pour",
                                "Box1","Bowl6","Box2","Bowl6","Box3","Bowl6","Box4","Bowl6","Box5","Bowl6",
                                "Bowl1", "Bowl6", "Region_Bw1", "Bowl2", "Bowl6", "Region_Bw2", "Bowl3", "Bowl6", "Region_Bw3",
                                "Bowl4", "Bowl6", "Region_Bw4", "Bowl5", "Bowl6", "Region_Bw5", "Bowl6", "Bowl7","Region_Bw6"]
        if data_type == 'action':
        
            #for order, mixing_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [32]
                else:
                    goal_state_num = [32, 29, 26, 23, 20, 17]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = mixing_withbox5.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = mixing_withbox5.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = mixing_withbox5.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = mixing_withbox5.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[mixing_action[i]])

                        target_object_index = node_id_to_idx[mixing_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "mixing_withbox5"
                        graph_dict_data['info']['goal'] = i_g
                        mixing_withbox5_graph.append(graph_dict_data)
            return mixing_withbox5_graph
        
    def collect_mixing_withbox4(self, data_type):
        mixing_withbox4_graph = []
        mixing_withbox4 = ReadDataset(task='mixing_withbox4', sort_edge=False)

        mixing_action = ['pick','place','pick','place',
                         'pick','place','pick','place','pick','place','pick','place',
                         'pick','pour','place','pick','pour','place','pick','pour','place',
                         'pick','pour','place','pick','pour','place']
        mixing_target_object = ["Bowl6","Region_Pour","Bowl7","Region_Pour",
                                "Box1","Bowl6","Box2","Bowl6","Box3","Bowl6","Box4","Bowl6",
                                "Bowl1", "Bowl6", "Region_Bw1", "Bowl2", "Bowl6", "Region_Bw2", "Bowl3", "Bowl6", "Region_Bw3",
                                "Bowl4", "Bowl6", "Region_Bw4", "Bowl6", "Bowl7","Region_Bw6"]
        if data_type == 'action':
        
            #for order, mixing_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [27]
                else:
                    goal_state_num = [27, 24, 21, 18, 15]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = mixing_withbox4.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = mixing_withbox4.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = mixing_withbox4.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = mixing_withbox4.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[mixing_action[i]])

                        target_object_index = node_id_to_idx[mixing_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "mixing_withbox4"
                        graph_dict_data['info']['goal'] = i_g
                        mixing_withbox4_graph.append(graph_dict_data)
            return mixing_withbox4_graph

        
    def collect_mixing_withbox3(self, data_type):
        mixing_withbox3_graph = []
        mixing_withbox3 = ReadDataset(task='mixing_withbox3', sort_edge=False)

        mixing_action = ['pick','place','pick','place',
                         'pick','place','pick','place','pick','place',
                         'pick','pour','place','pick','pour','place','pick','pour','place',
                         'pick','pour','place']
        mixing_target_object = ["Bowl6","Region_Pour","Bowl7","Region_Pour",
                                "Box1","Bowl6","Box2","Bowl6","Box3","Bowl6",
                                "Bowl1", "Bowl6", "Region_Bw1", "Bowl2", "Bowl6", "Region_Bw2", "Bowl3", "Bowl6", "Region_Bw3",
                                "Bowl6", "Bowl7","Region_Bw6"]
        if data_type == 'action':
        
            #for order, mixing_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [22]
                else:
                    goal_state_num = [22, 19, 16, 13]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = mixing_withbox3.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = mixing_withbox3.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = mixing_withbox3.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = mixing_withbox3.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[mixing_action[i]])

                        target_object_index = node_id_to_idx[mixing_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "mixing_withbox3"
                        graph_dict_data['info']['goal'] = i_g
                        mixing_withbox3_graph.append(graph_dict_data)
            return mixing_withbox3_graph

    def collect_mixing_withbox2(self, data_type):
        mixing_withbox2_graph = []
        mixing_withbox2 = ReadDataset(task='mixing_withbox2', sort_edge=False)

        mixing_action = ['pick','place','pick','place',
                         'pick','place','pick','place',
                         'pick','pour','place','pick','pour','place','pick','pour','place']
        mixing_target_object = ["Bowl6","Region_Pour","Bowl7","Region_Pour",
                                "Box1","Bowl6","Box2","Bowl6",
                                "Bowl1", "Bowl6", "Region_Bw1", "Bowl2", "Bowl6", "Region_Bw2", "Bowl6", "Bowl7","Region_Bw6"]
        if data_type == 'action':
        
            #for order, mixing_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [17]
                else:
                    goal_state_num = [17, 14, 11]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = mixing_withbox2.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = mixing_withbox2.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = mixing_withbox2.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = mixing_withbox2.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[mixing_action[i]])

                        target_object_index = node_id_to_idx[mixing_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "mixing_withbox2"
                        graph_dict_data['info']['goal'] = i_g
                        mixing_withbox2_graph.append(graph_dict_data)
            return mixing_withbox2_graph

    def collect_cleaning_box(self, data_type):
        cleaning_box_graph = []
        cleaning_box = ReadDataset(task='cleaning_box', sort_edge=False)

        cleaning_action = ['pick','place','pick','place','pick','place',
                         'pick','place','pick','place','pick','place',
                         'pick','place','pick','place','pick','place']
        cleaning_target_object = ["Box1","Region_Clean","Box2","Region_Clean","Box3","Region_Clean",
                                  "Box4","Region_Clean","Box5","Region_Clean","Box6","Region_Clean",
                                  "Box7","Region_Clean","Box8","Region_Clean","Box9","Region_Clean"]
        if data_type == 'action':
        
            #for order, cleaning_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [18]
                else:
                    goal_state_num = [18, 16, 14, 12, 10, 8, 6, 4, 2]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = cleaning_box.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = cleaning_box.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = cleaning_box.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = cleaning_box.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[cleaning_action[i]])

                        target_object_index = node_id_to_idx[cleaning_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "cleaning_box"
                        graph_dict_data['info']['goal'] = i_g
                        cleaning_box_graph.append(graph_dict_data)
            return cleaning_box_graph
    def collect_cleaning_init2(self, data_type):
        cleaning_init2_graph = []
        cleaning_init2 = ReadDataset(task='cleaning_init2', sort_edge=False)

        cleaning_action = ['pick','place','pick','place','pick','place',
                         'pick','place','pick','place','pick','place',
                         'pick','place']
        cleaning_target_object = ["Box6","Region_Clean","Box7","Region_Clean","Box8","Region_Clean",
                                  "Box9","Region_Clean","Box1","Region_Clean","Box2","Region_Clean",
                                  "Box3","Region_Clean"]
        if data_type == 'action':
        
            #for order, cleaning_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [14]
                else:
                    goal_state_num = [14, 12, 10, 8, 6, 4, 2]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = cleaning_init2.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = cleaning_init2.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = cleaning_init2.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = cleaning_init2.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[cleaning_action[i]])

                        target_object_index = node_id_to_idx[cleaning_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "cleaning_init2"
                        graph_dict_data['info']['goal'] = i_g
                        cleaning_init2_graph.append(graph_dict_data)
            return cleaning_init2_graph

    def collect_cleaning_init3(self, data_type):
        cleaning_init3_graph = []
        cleaning_init3 = ReadDataset(task='cleaning_init3', sort_edge=False)

        cleaning_action = ['pick','place','pick','place','pick','place',
                         'pick','place','pick','place','pick','place']
        cleaning_target_object = ["Box6","Region_Clean","Box7","Region_Clean","Box8","Region_Clean",
                                  "Box9","Region_Clean","Box1","Region_Clean","Box2","Region_Clean"]
        if data_type == 'action':
        
            #for order, cleaning_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [12]
                else:
                    goal_state_num = [12, 10, 8, 6, 4, 2]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = cleaning_init3.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = cleaning_init3.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = cleaning_init3.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = cleaning_init3.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[cleaning_action[i]])

                        target_object_index = node_id_to_idx[cleaning_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "cleaning_init3"
                        graph_dict_data['info']['goal'] = i_g
                        cleaning_init3_graph.append(graph_dict_data)
            return cleaning_init3_graph

    def collect_cleaning_init4(self, data_type):
        cleaning_init4_graph = []
        cleaning_init4 = ReadDataset(task='cleaning_init4', sort_edge=False)

        cleaning_action = ['pick','place','pick','place','pick','place',
                         'pick','place','pick','place']
        cleaning_target_object = ["Box6","Region_Clean","Box7","Region_Clean","Box8","Region_Clean",
                                  "Box9","Region_Clean","Box1","Region_Clean"]
        if data_type == 'action':
        
            #for order, cleaning_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [10]
                else:
                    goal_state_num = [10, 8, 6, 4, 2]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = cleaning_init4.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = cleaning_init4.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = cleaning_init4.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = cleaning_init4.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[cleaning_action[i]])

                        target_object_index = node_id_to_idx[cleaning_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "cleaning_init4"
                        graph_dict_data['info']['goal'] = i_g
                        cleaning_init4_graph.append(graph_dict_data)
            return cleaning_init4_graph

    def collect_cleaning_init5(self, data_type):
        cleaning_init5_graph = []
        cleaning_init5 = ReadDataset(task='cleaning_init5', sort_edge=False)

        cleaning_action = ['pick','place','pick','place','pick','place',
                         'pick','place']
        cleaning_target_object = ["Box6","Region_Clean","Box7","Region_Clean","Box8","Region_Clean",
                                  "Box9","Region_Clean"]
        if data_type == 'action':
        
            #for order, cleaning_target_object in order_seq_dict.items():
            for i in range(5):
                order = 'pose_'+str(i+1)
                if self.OnlyFinalGoal:
                    goal_state_num = [8]
                else:
                    goal_state_num = [8, 6, 4, 2]
                    
                for i_g in goal_state_num:
                    goal_node_feature, node_id_to_idx = cleaning_init5.node_features(order=order, i=i_g)
                    goal_edge_index, goal_edge_attr = cleaning_init5.edge_features(order=order, i=i_g)

                    for i in range(i_g):
                        x, _ = cleaning_init5.node_features(order=order, i=i)
                        state_edge_index, state_edge_attr = cleaning_init5.edge_features(order=order, i=i)

                        cat_edge_index, cat_edge_attr = self.concat_state_and_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)
                        action_code = torch.Tensor(self.action_encoder[cleaning_action[i]])

                        target_object_index = node_id_to_idx[cleaning_target_object[i]]
                        node_num = len(x)
                        target_object_score = np.zeros(node_num, dtype=int)
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
                        graph_dict_data['info']['demo'] = "cleaning_init5"
                        graph_dict_data['info']['goal'] = i_g
                        cleaning_init5_graph.append(graph_dict_data)
            return cleaning_init5_graph
               
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

stack_mix_poseX_data = CollectGraph()
action_data = stack_mix_poseX_data.collect_graph(data_type='action', dataset_name='stack_mix_clean_pose5',auto_save=True, OnlyFinalGoal=False)
print("num of data:{}".format(len(action_data)))
