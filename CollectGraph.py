import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import subgraph
import pickle

## Dataset
class ReadDataset():
    def __init__(self, task):
        # Search path
        # task: stacking / mixing / ...
        self.search_path = os.path.join(os.getcwd(), 'seq_dataset',task)

        print("search_path:",self.search_path)
            
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
    
#mix = ReadDataset('mixing_5')
#ex = mix.graph_data(order='1_2_3_5_4', i=3)

#print(ex['x'].shape, ex['edge_index'].shape, ex['edge_attr'].shape)






    


class CollectGraph():
    def __init__(self, goal_next=False, key_node=False, ):
        self.goal_next = goal_next
        self.action_encoder = {'pick':[1, 0, 0], 'place':[0, 1, 0], 'pour':[0, 0, 1]}

        #key_node check
        #concat check
            #edge_attr order check
        #fully_connected check? (implement)
        

    def collect_graph(self):
        collect_graph = []

        collect_graph.append(self.collect_stacking())
        collect_graph.append(self.collect_mixing())


    def collect_stacking(self):
        stacking_graph = []
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
                
                #cat_edge_index, cat_edge_attr = self.concat_state_n_goal(state_edge_index, state_edge_attr, goal_edge_index, goal_edge_attr)

        
    


def stacking_5_dataset():
    stacking_dataset = []

    make_data = CollectGraph(root_path=os.path.join('seq_dataset', 'stacking_5'))
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