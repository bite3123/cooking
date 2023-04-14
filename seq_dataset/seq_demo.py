import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
# from torch_geometric.data import Dataset
import os
import natsort
import random
import networkx as nx
import matplotlib.pyplot as plt
import PIL
import re


### Checking pandas dataframe without limit numbers of columns or rows 
# pd.options.display.max_columns = None
pd.options.display.max_rows = None

################################ Creating Folders ##############################################

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            pass
    except OSError:
        print ('Error: Creating directory.'  +  directory)

################################ Making datasets ################################################
class MakeDataset(Dataset):
    def __init__(self, problem, example):
        # Search path
        FILEPATH, _ = os.path.split(os.path.realpath(__file__))
        search_path = os.path.join(FILEPATH, 'tasks', problem, example)
        createFolder(search_path)

        self.FILEPATH = FILEPATH
        self.search_path = search_path
        self.problem = problem # task 종류
        self.example = example # pose
      

        print("\n==========================================INIT======================================================")
        print("\n[File path]",FILEPATH)
        print("\n[Search_path]",search_path)
        print("\n[Example]", example)
        print("\n[Problem]", problem)
        print("\n==========================================INIT======================================================")
    
    # Call input file (Choose feature: 'node_features', 'edge_index', 'edge_attr')
    def input_csv_file(self, n, feature, index_col):    
        feature_path = os.path.join(self.search_path, feature)
        file_list = natsort.natsorted(os.listdir(feature_path))
        input_path = os.path.join(feature_path, file_list[n])
        input_df = pd.read_csv(input_path, index_col= index_col)
	        
        return input_df
    
    # Save output file (Choose feature: 'node_features', 'edge_index', 'edge_attr')
    def output_csv_file(self, n, features, which_csv):
        if features == 'node_features':
            file_name = f'{self.example}_nf{n}.csv'
        elif features == 'edge_index':
            file_name = f'{self.example}_ei{n}.csv'
        elif features == 'edge_attr':
            file_name = f'{self.example}_ea{n}.csv'
        else:
            print("----Check the file name again----")
        save_path = os.path.join(self.search_path, features)
        createFolder(save_path)
        which_csv.to_csv(os.path.join(save_path,file_name))

    # Start to make node feature file
    def init_node_features(self):
        save_path = os.path.join(self.search_path,'node_features')
        createFolder(save_path)

        if 'stacking' in self.problem:
            print("====[Stacking]====")
        elif 'mixing' in self.problem:
            print("====[Mixing]====")
        #ID_col = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Table']
        #28개
        ID_col = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Box6','Box7','Box8','Box9',
                    'Box_Bw1','Box_Bw2','Box_Bw3','Box_Bw4','Box_Bw5',
                    'Bowl1','Bowl2','Bowl3','Bowl4','Bowl5','Bowl6','Bowl7',
                    'Region_Bw1','Region_Bw2','Region_Bw3','Region_Bw4','Region_Bw5','Region_Bw6','Region_Bw7',
                    'Region_Stack', 'Region_Pour', 'Region_Free', 'Region_Clean']
        all_0 = [0]*len(ID_col)
        

        nf0 = pd.DataFrame({'ID': ID_col,'Type_Bowl': all_0, 'Type_Box': all_0, 'Type_Robot': all_0, \
                                        'Type_Region': all_0, 'Property_V': all_0, 'Property_G': all_0,\
                                        'Position_x': all_0, 'Position_y': all_0, 'Position_z': all_0, \
                                        'Orientation_x':all_0, 'Orientation_y':all_0, 'Orientation_z':all_0})
        nf0 = nf0.set_index("ID")

        box_order_x = [0.2, 0.3, 0.4]
        box_order_y = [0.05, 0.15, 0.25]
        np.random.shuffle(box_order_x)
        np.random.shuffle(box_order_y)

        for nf_inx in nf0.index:
            if "Box" in nf_inx:
                nf0.loc[nf_inx, 'Type_Box'] = 1 
                nf0.loc[nf_inx, 'Property_G'] = 1 
                # 랜덤하게 V 추가
                if random.random() >= 0.5:
                   nf0.loc[nf_inx, 'Property_V'] = 1
                if "Bw" in nf_inx:
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    if "Bw1" in nf_inx:
                        nf0.loc[nf_inx, 'Position_y'] = -0.1
                    elif "Bw2" in nf_inx:
                        nf0.loc[nf_inx, 'Position_y'] = -0.2             
                    elif "Bw3" in nf_inx:
                        nf0.loc[nf_inx, 'Position_y'] = -0.3
                    elif "Bw4" in nf_inx:
                        nf0.loc[nf_inx, 'Position_y'] = -0.4
                    elif "Bw5" in nf_inx:
                        nf0.loc[nf_inx, 'Position_y'] = -0.5     
                else:
                    box_idx = int(nf_inx[-1])
                    x_idx, y_idx = divmod(box_idx-1, 3)
                    rand_px, rand_py = np.random.uniform(-0.025, 0.025, 2)
                    rand_oz = np.random.uniform(-45, 45, 1)
                    nf0.loc[nf_inx, 'Position_x'] = box_order_x[x_idx] + rand_px
                    nf0.loc[nf_inx, 'Position_y'] = box_order_y[y_idx] + rand_py
                    nf0.loc[nf_inx, 'Orientation_z'] = rand_oz[0]


            elif "Bowl" in nf_inx:
                nf0.loc[nf_inx, 'Type_Bowl'] = 1 
                nf0.loc[nf_inx, 'Property_G'] = 1 
                if nf_inx == "Bowl1":
                    #h=0.08 w=0.08
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    nf0.loc[nf_inx, 'Position_y'] = -0.1
                elif nf_inx == "Bowl2":
                    #h=0.08 w=0.08
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    nf0.loc[nf_inx, 'Position_y'] = -0.2
                elif nf_inx == "Bowl3":
                    #h=0.08 w=0.08
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    nf0.loc[nf_inx, 'Position_y'] = -0.3       
                elif nf_inx == "Bowl4":
                    #h=0.08 w=0.08
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    nf0.loc[nf_inx, 'Position_y'] = -0.4
                elif nf_inx == "Bowl5":
                    #h=0.08 w=0.08
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    nf0.loc[nf_inx, 'Position_y'] = -0.5
                elif nf_inx == "Bowl6":
                    #h=0.1 w=0.1
                    nf0.loc[nf_inx, 'Position_x'] = -0.1
                    nf0.loc[nf_inx, 'Position_y'] = -0.4
                elif nf_inx == "Bowl7":
                    #h=0.1 w=0.1
                    nf0.loc[nf_inx, 'Position_x'] = -0.1
                    nf0.loc[nf_inx, 'Position_y'] = -0.5
            elif "Robot" in nf_inx:
                nf0.loc[nf_inx, 'Type_Robot'] = 1 
                nf0.loc[nf_inx, 'Position_z'] = +0.5 
            elif "Region" in nf_inx:
                nf0.loc[nf_inx, 'Type_Region'] = 1
                if nf_inx == "Region_Bw1":
                    #h=0.08 w=0.08
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    nf0.loc[nf_inx, 'Position_y'] = -0.1
                elif nf_inx == "Region_Bw2":
                    #h=0.08 w=0.08
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    nf0.loc[nf_inx, 'Position_y'] = -0.2
                elif nf_inx == "Region_Bw3":
                    #h=0.08 w=0.08
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    nf0.loc[nf_inx, 'Position_y'] = -0.3       
                elif nf_inx == "Region_Bw4":
                    #h=0.08 w=0.08
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    nf0.loc[nf_inx, 'Position_y'] = -0.4
                elif nf_inx == "Region_Bw5":
                    #h=0.08 w=0.08
                    nf0.loc[nf_inx, 'Position_x'] = -0.2
                    nf0.loc[nf_inx, 'Position_y'] = -0.5
                elif nf_inx == "Region_Bw6":
                    #h=0.1 w=0.1
                    nf0.loc[nf_inx, 'Position_x'] = -0.1
                    nf0.loc[nf_inx, 'Position_y'] = -0.4
                elif nf_inx == "Region_Bw7":
                    #h=0.1 w=0.1
                    nf0.loc[nf_inx, 'Position_x'] = -0.1
                    nf0.loc[nf_inx, 'Position_y'] = -0.5
                elif nf_inx == "Region_Stack":
                    #h=0.3 w=0.3
                    nf0.loc[nf_inx, 'Position_x'] = +0.15
                    nf0.loc[nf_inx, 'Position_y'] = +0.3
                elif nf_inx == "Region_Pour":
                    #h=0.3 w=0.3
                    nf0.loc[nf_inx, 'Position_x'] = +0.15
                    nf0.loc[nf_inx, 'Position_y'] = -0.3
                elif nf_inx == "Region_Free":
                    #h=0.3 w=0.3
                    nf0.loc[nf_inx, 'Position_x'] = +0.15
                elif nf_inx == "Region_Clean":
                    #h=0.3 w=0.3
                    nf0.loc[nf_inx, 'Position_x'] = -0.15
                    nf0.loc[nf_inx, 'Position_y'] = +0.45

        
        #task에 따라서 initial scene 조정
        if "init2" in self.problem:
            #박스 2개 stack된 상태로 나머지 3개 stacking
            nf0.loc['Box4', 'Position_x':'Orientation_z'] = nf0.loc['Box5', 'Position_x':'Orientation_z']
            nf0.loc['Box4', 'Position_z'] += 0.04
        #elif self.problem == "stacking_init3" or self.problem == "stacking_init3_replace" or self.problem == "stacking_init3_reverse":
        elif "init3" in self.problem:
            #stacking_init3: 박스 3개 stack된 상태로 나머지 2개 stacking
            #stacking_init3_replace: 박스 3개 stack된 상태에서 중간 박스를 다른 박스로 교체
            nf0.loc['Box4', 'Position_x':'Orientation_z'] = nf0.loc['Box5', 'Position_x':'Orientation_z']
            nf0.loc['Box4', 'Position_z'] += 0.04
            nf0.loc['Box3', 'Position_x':'Orientation_z'] = nf0.loc['Box4', 'Position_x':'Orientation_z']
            nf0.loc['Box3', 'Position_z'] += 0.04

        elif "init4" in self.problem:
            nf0.loc['Box4', 'Position_x':'Orientation_z'] = nf0.loc['Box5', 'Position_x':'Orientation_z']
            nf0.loc['Box4', 'Position_z'] += 0.04
            nf0.loc['Box3', 'Position_x':'Orientation_z'] = nf0.loc['Box4', 'Position_x':'Orientation_z']
            nf0.loc['Box3', 'Position_z'] += 0.04
            nf0.loc['Box2', 'Position_x':'Orientation_z'] = nf0.loc['Box3', 'Position_x':'Orientation_z']
            nf0.loc['Box2', 'Position_z'] += 0.04
        elif "init5" in self.problem:

            nf0.loc['Box4', 'Position_x':'Orientation_z'] = nf0.loc['Box5', 'Position_x':'Orientation_z']
            nf0.loc['Box4', 'Position_z'] += 0.04
            nf0.loc['Box3', 'Position_x':'Orientation_z'] = nf0.loc['Box4', 'Position_x':'Orientation_z']
            nf0.loc['Box3', 'Position_z'] += 0.04
            nf0.loc['Box2', 'Position_x':'Orientation_z'] = nf0.loc['Box3', 'Position_x':'Orientation_z']
            nf0.loc['Box2', 'Position_z'] += 0.04
            nf0.loc['Box1', 'Position_x':'Orientation_z'] = nf0.loc['Box2', 'Position_x':'Orientation_z']
            nf0.loc['Box1', 'Position_z'] += 0.04
        else:
            pass
        
        self.output_csv_file(0, 'node_features', nf0)
        print(nf0)

    # Hierarchical에서 node feauture의 정보가 바뀔 때
    # def changed_node_feature(self, n):
    #     nf0 = self.input_csv_file(n-1, 'node_features', None)
    #     # print(nf0)
    #     attach_i = nf0[nf0['Property_V'] == 1]
    #     attach_index = attach_i.index.to_list() #[int, int]
    #     attach_id = attach_i["ID"].to_list()
    #     # print(attach_index)
    
    #     attach_boxes = str(attach_id[-2]) + str(attach_id[-1])
        
    #     # nf0 = nf0.drop(index=attach_index[-1])
    #     # nf0 = nf0.drop(index=attach_index[-2])
    #     nf1 = nf0.set_index("ID")
    #     nf1.loc[attach_boxes] = [0 for i in range(len(nf1.columns))]
    
    #     list_0 = [0 for n in range(len(nf1.index)-1)] + [1]
    #     nf1.loc[:,'Type_Attached_Boxes'] = list_0
    #     nf1.loc[attach_boxes, 'Property_V'] = 1
    #     nf1.loc[attach_boxes, 'Property_G'] = 1

    #     print(f'\n[{self.example}_nf{n}]\n',nf1)
        
    #     self.output_csv_file(n, 'node_features', nf1)
    #     # save_csv = os.path.join(nf_path, f'{self.example}_nf{n}.csv')
    #     # nf1.to_csv(save_csv) 
        
    #     print(f'\n----{self.example}_nf{n}.csv file is saved----')

    # Copy the dataframe from init_node_features()
    def same_node_features(self, n):

        nf = self.input_csv_file(n-1, 'node_features', None)
        nf = nf.set_index("ID")
        self.output_csv_file(n, 'node_features', nf)
       
        # print(nf)
        print(f'\n----{self.example}_nf{n}.csv file is saved----')
    
    def pick_nf(self, n, obj1):
        #obj1: graspable object

        nf = self.input_csv_file(n-1, 'node_features', 0)

        nf.loc[obj1, 'Position_x':'Orientation_z'] = nf.loc['Robot_hand','Position_x':'Orientation_z']

        if 'Bowl' in obj1:
            bowl_csv = self.input_csv_file(n-1, 'edge_index', 0)

            bowl_objects = bowl_csv.index
            # print('Pour object', pour_objects)
            # Node feature 따라서 만들고 objects은 bowl이어야 함
            for bowl_obj in bowl_objects:
                if 'Box' in bowl_obj:
                    if (bowl_csv.at[obj1, bowl_obj] == 1) and (bowl_csv.at[bowl_obj, obj1] == 1):
                        nf.loc[bowl_obj, 'Position_x':'Orientation_z'] = nf.loc['Robot_hand','Position_x':'Orientation_z']
        

        self.output_csv_file(n, 'node_features', nf)
       
        # print(nf)
        print(f'\n----{self.example}_nf{n}.csv file is saved----')

    def place_nf(self, n, obj1, obj2):
        #obj1: grasped object in robot_hand
        #obj2: box-> stack / bowl -> in / region -> place

        nf = self.input_csv_file(n-1, 'node_features', 0)

        nf.loc[obj1, 'Position_x':'Orientation_z'] = nf.loc[obj2, 'Position_x':'Orientation_z']
        if 'Box' in obj2:
            nf.loc[obj1,'Position_z'] += 0.04
        elif 'Region' in obj2:
            if 'Bowl' in obj1:
                if 'Pour' in obj2:
                    rand_px, rand_py = np.random.uniform(-0.15, 0.15, 2)
                    nf.loc[obj1,'Position_x'] += rand_px
                    nf.loc[obj1,'Position_y'] += rand_py
            #elif ('Stack' in obj2) or ('Pour' in obj2):
            elif 'Box' in obj1:
                rand_px, rand_py = np.random.uniform(-0.15, 0.15, 2)
                rand_oz = np.random.uniform(-45, 45, 1)
                nf.loc[obj1,'Position_x'] += rand_px
                nf.loc[obj1,'Position_y'] += rand_py
                nf.loc[obj1,'Orientation_z'] += rand_oz[0]
        elif 'Bowl' in obj2:
            rand_px, rand_py = np.random.uniform(-0.05, 0.05, 2)
            rand_pz = np.random.uniform(0, 0.1, 1)
            rand_ox, rand_oy, rand_oz = np.random.uniform(-180, 180, 3)
            nf.loc[obj1,'Position_x'] += rand_px
            nf.loc[obj1,'Position_y'] += rand_py
            nf.loc[obj1,'Position_z'] += rand_pz[0]
            nf.loc[obj1,'Orientation_x'] += rand_ox
            nf.loc[obj1,'Orientation_y'] += rand_oy
            nf.loc[obj1,'Orientation_z'] += rand_oz

        self.output_csv_file(n, 'node_features', nf)

        # print(nf)
        print(f'\n----{self.example}_nf{n}.csv file is saved----')

    def pour_nf(self, n, obj1, obj2):
        #obj1, obj2: bowls

        nf = self.input_csv_file(n-1, 'node_features', 0)
        pour_csv = self.input_csv_file(n-1, 'edge_index', 0)

        pour_objects = pour_csv.index
        # print('Pour object', pour_objects)
        # Node feature 따라서 만들고 objects은 bowl이어야 함
        for pour_obj in pour_objects:
            if 'Box' in pour_obj:
                if (pour_csv.at[obj1, pour_obj] == 1) and (pour_csv.at[pour_obj, obj1] == 1):
                    nf.loc[pour_obj, 'Position_x':'Orientation_z'] = nf.loc[obj2, 'Position_x':'Orientation_z']

                    rand_px, rand_py = np.random.uniform(-0.05, 0.05, 2)
                    rand_pz = np.random.uniform(0, 0.1, 1)
                    rand_ox, rand_oy, rand_oz = np.random.uniform(-180, 180, 3)
                    nf.loc[pour_obj,'Position_x'] += rand_px
                    nf.loc[pour_obj,'Position_y'] += rand_py
                    nf.loc[pour_obj,'Position_z'] += rand_pz[0]
                    nf.loc[pour_obj,'Orientation_x'] += rand_ox
                    nf.loc[pour_obj,'Orientation_y'] += rand_oy
                    nf.loc[pour_obj,'Orientation_z'] += rand_oz
        
        
        self.output_csv_file(n, 'node_features', nf)
       
        # print(nf)
        print(f'\n----{self.example}_nf{n}.csv file is saved----')


###### Edge index ######
    # Relationship of the 'Box & Table (without attached boxes)', 'Bowl & Table'     
    def edge_csv(self,ef0):
        for ef_inx in ef0.index:
            if "Box" in ef_inx:
                if 'Bw' in ef_inx:
                    ef0.loc[ef_inx, f'Bowl{ef_inx[-1]}'] = 1
                    ef0.loc[f'Bowl{ef_inx[-1]}', ef_inx] = 1
                else:
                    # Region_Stack
                    ef0.loc[ef_inx, 'Region_Stack'] = 1 
                    ef0.loc['Region_Stack', ef_inx] = 1
            elif "Bowl" in ef_inx:
                target_region = f'Region_Bw{str(ef_inx[-1])}'
                ef0.loc[ef_inx, target_region] = 1 
                ef0.loc[target_region, ef_inx] = 1  

            if self.problem == 'stacking_init2':
                ef0.loc['Box4', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box4'] = 0
                ef0.loc['Box4', 'Box5'] = 1
                ef0.loc['Box5', 'Box4'] = 1
            
            #elif self.problem == "stacking_init3" or self.problem == "stacking_init3_replace" or self.problem == "stacking_init3_reverse":
            elif "stacking_init3" in self.problem:
                ef0.loc['Box4', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box4'] = 0
                ef0.loc['Box4', 'Box5'] = 1
                ef0.loc['Box5', 'Box4'] = 1
                ef0.loc['Box3', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box3'] = 0
                ef0.loc['Box3', 'Box4'] = 1
                ef0.loc['Box4', 'Box3'] = 1
            elif self.problem == "cleaning_init5":
                ef0.loc['Box4', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box4'] = 0
                ef0.loc['Box4', 'Box5'] = 1
                ef0.loc['Box5', 'Box4'] = 1
                ef0.loc['Box3', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box3'] = 0
                ef0.loc['Box3', 'Box4'] = 1
                ef0.loc['Box4', 'Box3'] = 1
                ef0.loc['Box2', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box2'] = 0
                ef0.loc['Box2', 'Box3'] = 1
                ef0.loc['Box3', 'Box2'] = 1
                ef0.loc['Box1', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box1'] = 0
                ef0.loc['Box1', 'Box2'] = 1
                ef0.loc['Box2', 'Box1'] = 1
            elif self.problem == "cleaning_init4":
                ef0.loc['Box4', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box4'] = 0
                ef0.loc['Box4', 'Box5'] = 1
                ef0.loc['Box5', 'Box4'] = 1
                ef0.loc['Box3', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box3'] = 0
                ef0.loc['Box3', 'Box4'] = 1
                ef0.loc['Box4', 'Box3'] = 1
                ef0.loc['Box2', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box2'] = 0
                ef0.loc['Box2', 'Box3'] = 1
                ef0.loc['Box3', 'Box2'] = 1
            elif self.problem == "cleaning_init3":
                ef0.loc['Box4', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box4'] = 0
                ef0.loc['Box4', 'Box5'] = 1
                ef0.loc['Box5', 'Box4'] = 1
                ef0.loc['Box3', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box3'] = 0
                ef0.loc['Box3', 'Box4'] = 1
                ef0.loc['Box4', 'Box3'] = 1
            elif self.problem == "cleaning_init2":
                ef0.loc['Box4', 'Region_Stack'] = 0
                ef0.loc['Region_Stack', 'Box4'] = 0
                ef0.loc['Box4', 'Box5'] = 1
                ef0.loc['Box5', 'Box4'] = 1   
            else:
                pass


        print("[Init edge index]\n",ef0)

    # Save problem as a dataframe of edge index
    def make_problem(self):
        nf = self.input_csv_file(0, 'node_features', None)
        node_index = nf['ID'].to_list()
        ef0 = pd.DataFrame(0, index= node_index, columns= node_index)
        ef0.index.name = "ID"
        self.edge_csv(ef0)
        self.output_csv_file(0, 'edge_index', ef0)
    
    # Make initial edge index
    def init_edge_index(self):
        self.make_problem()

        
    # Pick edge index
    def pick_inx(self, n, obj1): # obj = ID number
        # Choose sample
        pick_csv = self.input_csv_file(n-1, 'edge_index', 0)#이전스텝 불러와서 업데이트하는 방식


        # Remove 'on' relation (Table / other box)
        if 'Box' in obj1:
            pick_csv.loc[obj1,:] = 0
            pick_csv.loc[:,obj1] = 0
        elif 'Bowl' in obj1:
            #target_region = f'Region_Bw{str(obj1[-1])}'
            pick_csv.loc[obj1,15:] = 0
            pick_csv.loc[15:,obj1] = 0

        # Add 'in-grasp' relation (Robot-hand)
        pick_csv.loc[obj1,'Robot_hand'] = 1
        pick_csv.loc['Robot_hand',obj1] = 1
        # Save files
        self.output_csv_file(n, 'edge_index', pick_csv)
    
        print(f'\n[Pick[{str(obj1)}].csv] \n') 
        return pick_csv


    # Place edge index
    def place_inx(self, n, obj1, obj2): 
        place_csv = self.input_csv_file(n-1, 'edge_index', 0)

        # Add relation with obj1 and obj2
        place_csv.loc[obj1,obj2] = 1
        place_csv.loc[obj2,obj1] = 1

        # Remove 'in-grasp' relation (Robot hand)
        place_csv.loc[obj1,'Robot_hand'] = 0
        place_csv.loc['Robot_hand',obj1] = 0
    

        ### Save files
        self.output_csv_file(n, 'edge_index', place_csv)

        print(f'\n[Place_[{str(obj1)}]_on_[{str(obj2)}].csv] \n') 
        return place_csv
            

    # Pour edge index
    def pour_inx(self,n, obj1, obj2):
        pour_csv = self.input_csv_file(n-1, 'edge_index', 0)
        pour_objects = pour_csv.index
        # print('Pour object', pour_objects)
        # Node feature 따라서 만들고 objects은 bowl이어야 함
        for pour_obj in pour_objects:
            if 'Box' in pour_obj:
                if (pour_csv.at[obj1, pour_obj] == 1) and (pour_csv.at[pour_obj, obj1] == 1):
                    # print("[[Pour obj]]", pour_obj)
                    pour_csv.at[obj1, pour_obj] = 0
                    pour_csv.at[pour_obj, obj1] = 0
                    pour_csv.at[obj2, pour_obj] = 1
                    pour_csv.at[pour_obj, obj2] = 1

        print(f'\n[Pour_[{str(obj1)}]_on_[{str(obj2)}].csv] \n') 
        ### Save files
        self.output_csv_file(n, 'edge_index', pour_csv)
        return pour_csv
    

###### Edge attr ######
    # Make initial edge attribute file
    def init_edge_attr(self,num):
        list_attr = []
        list_attr1 = []
        list_attr_0 = []

        # Dataframe
        for i in range(num+1):   
            ef = self.input_csv_file(i, 'edge_index', 0)
            ID_list = list(map(str, ef.columns))
            for index in range(len(ID_list)):
                for column in range(len(ID_list)):
                    if ef.iat[index, column] == 1:  
                        list_attr1.append((ID_list[index], ID_list[column]))
                        if i==0:
                            list_attr_0.append((ID_list[index], ID_list[column]))
        # 자체 중복 방지
        for v in list_attr1:
            if v not in list_attr:
                list_attr.append(v)
       


        list_0 = [0 for i in range(len(list_attr))]
        

        # print(list_attr, "length", len(list_attr))
        edge_attr0_csv = pd.DataFrame({'ID': list_attr, 'rel_on_right':list_0, 'rel_on_left': list_0, \
                                        'rel_in_right':list_0, 'rel_in_left': list_0, 
                                        'rel_in_grasp':list_0, 'rel_grasp': list_0, 'rel_attach': list_0 })
                                        #'pos_x': list_0, 'pos_y': list_0, 'pos_z': list_0, \
                                        #'pos_roll':list_0, 'pos_pitch':list_0, 'pos_yaw':list_0})
        edge_attr0_csv = edge_attr0_csv.set_index("ID")
        
        # print(list_attr1)
        for node_pair in list_attr_0:
            np0 = node_pair[0]
            np1 = node_pair[1]
            if 'Region_Stack' in np0 and 'Box' in np1:
                edge_attr0_csv.loc[[node_pair], 'rel_on_left'] = 1
            elif 'Region_Stack' in np1 and 'Box' in np0:
                edge_attr0_csv.loc[[node_pair], 'rel_on_right'] = 1

            elif np0[-1] == np1[-1]:
                if 'Box' in np0 and 'Bowl' in np1 :
                    edge_attr0_csv.loc[[node_pair], 'rel_in_right'] = 1
                elif 'Bowl' in np0 and 'Box' in np1:
                    edge_attr0_csv.loc[[node_pair], 'rel_in_left'] = 1
                elif 'Region' in np0 and 'Bowl' in np1:
                    edge_attr0_csv.loc[[node_pair], 'rel_on_left'] = 1
                elif 'Bowl' in np0 and 'Region' in np1:
                    edge_attr0_csv.loc[[node_pair], 'rel_on_right'] = 1
            else:
                print("wrong")

        if "init2" in self.problem:
            edge_attr0_csv.loc[[('Box4', 'Box5')], 'rel_on_right'] = 1
            edge_attr0_csv.loc[[('Box5', 'Box4')], 'rel_on_left'] = 1

        #elif self.problem == "stacking_init3" or self.problem == "stacking_init3_replace" or self.problem == "stacking_init3_reverse":
        elif "init3" in self.problem:
            edge_attr0_csv.loc[[('Box4', 'Box5')], 'rel_on_right'] = 1
            edge_attr0_csv.loc[[('Box5', 'Box4')], 'rel_on_left'] = 1
            edge_attr0_csv.loc[[('Box3', 'Box4')], 'rel_on_right'] = 1
            edge_attr0_csv.loc[[('Box4', 'Box3')], 'rel_on_left'] = 1
        elif "init4" in self.problem:
            edge_attr0_csv.loc[[('Box4', 'Box5')], 'rel_on_right'] = 1
            edge_attr0_csv.loc[[('Box5', 'Box4')], 'rel_on_left'] = 1
            edge_attr0_csv.loc[[('Box3', 'Box4')], 'rel_on_right'] = 1
            edge_attr0_csv.loc[[('Box4', 'Box3')], 'rel_on_left'] = 1
            edge_attr0_csv.loc[[('Box2', 'Box3')], 'rel_on_right'] = 1
            edge_attr0_csv.loc[[('Box3', 'Box2')], 'rel_on_left'] = 1
        elif "init5" in self.problem:
            edge_attr0_csv.loc[[('Box4', 'Box5')], 'rel_on_right'] = 1
            edge_attr0_csv.loc[[('Box5', 'Box4')], 'rel_on_left'] = 1
            edge_attr0_csv.loc[[('Box3', 'Box4')], 'rel_on_right'] = 1
            edge_attr0_csv.loc[[('Box4', 'Box3')], 'rel_on_left'] = 1
            edge_attr0_csv.loc[[('Box2', 'Box3')], 'rel_on_right'] = 1
            edge_attr0_csv.loc[[('Box3', 'Box2')], 'rel_on_left'] = 1
            edge_attr0_csv.loc[[('Box1', 'Box2')], 'rel_on_right'] = 1
            edge_attr0_csv.loc[[('Box2', 'Box1')], 'rel_on_left'] = 1
        # SAVE PATH (edge_attr)
        self.output_csv_file(0, 'edge_attr', edge_attr0_csv)
                    
        print("\n[init_ea0.csv]\n",edge_attr0_csv)        
        print("\n----Edge attribute is saved----")

    ## Tuple로 list가 들어간 경우에는 따옴표가 이중으로 들어가버림 -> 따옴표 없애는 용도
    def list_changer(self, edge_attr_i):
        new_list = []
        for item in edge_attr_i:
            new_item = tuple(map(lambda x: x.strip("' "), item.strip("()").split(",")))
            new_list.append(new_item)
        # print("nn",new_list)
        return new_list

    # Pick edge attribute
    def pick_attr(self, n, obj1):
        edge_attr_csv = self.input_csv_file(n-1, 'edge_attr', 0)
        edge_attr_index = edge_attr_csv.index.to_list() 
        new_list = self.list_changer(edge_attr_index)
        

        for npr in new_list:           
            np0 = npr[0]
            np1 = npr[1]
            npa = (np0, np1)
            node_pair= "('{}', '{}')".format(*npa)

            # if 'stacking' in self.problem:
            # Delete previous relationship
            if np0 == obj1:
                if 'Bowl' in np0 and 'Region' in np1:
                    edge_attr_csv.loc[[node_pair], :] = 0
                elif 'Box' in np0:
                    edge_attr_csv.loc[[node_pair], :] = 0
                if np1 == 'Robot_hand':
                    edge_attr_csv.loc[[node_pair], 'rel_in_grasp'] = 1 
            elif np1 == obj1:
                if 'Bowl' in np1 and 'Region' in np0:
                    edge_attr_csv.loc[[node_pair], :] = 0
                elif 'Box' in np1:
                    edge_attr_csv.loc[[node_pair], :] = 0
                if np0 == 'Robot_hand':
                    edge_attr_csv.loc[[node_pair], 'rel_grasp'] = 1 

            '''
            if (np0 == obj1) and ('Region' in np1):
                edge_attr_csv.loc[[node_pair], :] = 0
            elif (np1 == obj1) and ('Region' in np0):
                edge_attr_csv.loc[[node_pair], :] = 0

            # Grasp 'obj1 in robot-hand
            elif (np0 == 'Robot_hand') and (np1 == obj1):
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[node_pair], 'rel_grasp'] = 1
            elif (np1 == 'Robot_hand') and (np0 == obj1):
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[node_pair], 'rel_in_grasp'] = 1 
            '''
    
        print(f"\n[Pick[{obj1}].csv]\n",edge_attr_csv)

        # # SAVE PATH (edge_attr)
        self.output_csv_file(n, 'edge_attr', edge_attr_csv)
        print(f"\n----{self.example}_ea{n}.csv is saved----")

    # Place edge attribute
    def place_attr(self, n, obj1, obj2):

        edge_attr_csv = self.input_csv_file(n-1,'edge_attr',0)
        edge_attr_index = edge_attr_csv.index.to_list()
        new_list = self.list_changer(edge_attr_index)
        for npr in new_list:           
            np0 = npr[0]
            np1 = npr[1]
            npa = (np0, np1)
            node_pair= "('{}', '{}')".format(*npa)
            
            if np0 == obj1:
                if np1 == 'Robot_hand':
                    edge_attr_csv.loc[[node_pair], :] = 0
                elif np1 == obj2:
                    if 'Region' in np1 or 'Box' in np1:
                        edge_attr_csv.loc[[node_pair], 'rel_on_right'] = 1
                    else:
                        edge_attr_csv.loc[[node_pair], 'rel_in_right'] = 1
            elif np1 == obj1:
                if np0 == 'Robot_hand':
                    edge_attr_csv.loc[[node_pair], :] = 0
                elif np0 == obj2:
                    if 'Region' in np0 or 'Box' in np0:
                        edge_attr_csv.loc[[node_pair], 'rel_on_left'] = 1
                    else:
                        edge_attr_csv.loc[[node_pair], 'rel_in_left'] = 1      
            '''
            # if 'stacking' in self.problem:
            # Delete previous relationship
            if (np0 == 'Robot_hand') and (np1 == obj1):
                edge_attr_csv.loc[[node_pair], :] = 0
            elif (np1 == 'Robot_hand') and (np0 == obj1):
                edge_attr_csv.loc[[node_pair], :] = 0

            # Obj1 is on-relationship with Obj2
            elif np0 == obj1 and np1 == obj2:
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[node_pair], 'rel_on_right'] = 1
            elif np0 == obj2 and np1 == obj1:
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[node_pair], 'rel_on_left'] = 1   
            '''
        print(f"\n[Place[{obj1}]_on_[{obj2}].csv]\n",edge_attr_csv)

        # SAVE PATH (edge_attr)
        self.output_csv_file(n, 'edge_attr', edge_attr_csv)
        print(f"\n----{self.example}_ea{n}.csv is saved----")




    # Pour edge attribute
    def pour_attr(self, n, obj1, obj2):
        edge_attr_csv = self.input_csv_file(n-1,'edge_attr',0)
        edge_attr_index = edge_attr_csv.index.to_list()
        new_list = self.list_changer(edge_attr_index)

        for npr in new_list:           
            np0 = npr[0]
            np1 = npr[1]
            npa = (np0, np1)
            node_pair= "('{}', '{}')".format(*npa)
            if np0 == obj1 and 'Box' in np1:#np1: obj1에 담겨있는 물체
                # print(node_pair)
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[f"('{obj2}', '{np1}')"], 'rel_in_left'] = 1
                # edge_attr_csv.loc[[f"('{np1}', '{obj2}')"], 'rel_in_left'] = 5
                

            if np1 == obj1 and 'Box' in np0:#np0: obj1에 담겨있는 물체
                # print(node_pair)
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[f"('{np0}', '{obj2}')"], 'rel_in_right'] = 1
        print(f"\n[Pour[{obj1}]_to_[{obj2}].csv]\n",edge_attr_csv)

        # SAVE PATH (edge_attr)
        self.output_csv_file(n, 'edge_attr', edge_attr_csv)
        print(f"\n----{self.example}_ea{n}.csv is saved----")
