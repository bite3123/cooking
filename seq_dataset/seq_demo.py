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
        self.problem = problem
        self.example = example
      

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


    # Search boxes which has 'Property_V' from node feature file
    def df_attach_v2(self,nf):
        attach_v = nf[nf['Property_V'] == 1].index.to_list()
        # print("\nAttach ID",attach_id)
        
        attach_boxes = str(attach_v[-2]) + str(' + ')+ str(attach_v[-1])
        nf.loc[attach_boxes] = [0]*len(nf.columns)
        nf.loc[:,'Type_Attached_Boxes'] = [0]*(len(nf.index)-1) + [1]
        nf.loc[attach_boxes, ['Property_V', 'Property_G']] = 1
        self.attach_v = attach_v

    # Search boxes which has 'Property_V' from node feature file
    def df_attach_v3(self,nf):
        # print("\n[Middle]\n",nf)
        attach_v = nf[nf['Property_V'] == 1].index.to_list()
        left_id = [x for x in attach_v if x not in self.attach_v[-2] and x not in self.attach_v[-1]]

        attach_boxes = str(left_id[-2]) + str(' + ')+ str(left_id[-1])
        # 나중에 hierarchy 만들 때 drop해서 만들 수 있음
        # nf0 = nf0.drop(index=attach_id[-1])
        # nf0 = nf0.drop(index=attach_id[-2])
        # print(nf1)
        nf.loc[attach_boxes] = [0]*len(nf.columns)
        nf.loc[attach_boxes,['Type_Attached_Boxes','Property_V','Property_G']] = 1 
        self.left_v3 = left_id

    # Search boxes which has 'Property_V' from node feature file       
    def df_attach_v4(self,nf):
        # print("\n[Middle]\n",nf)
        attach_v = nf[nf['Property_V'] == 1].index.to_list()
        left_id = [x for x in attach_v if x not in self.left_v3[-2] and x not in self.left_v3[-1]]

        attach_boxes = str(left_id[-2]) + str(' + ')+ str(left_id[-1])
        # nf0 = nf0.drop(index=attach_id[-1])
        # nf0 = nf0.drop(index=attach_id[-2])
        # print(nf1)
        nf.loc[attach_boxes] = [0]*len(nf.columns)
        nf.loc[attach_boxes,['Type_Attached_Boxes','Property_V','Property_G']] = 1    



    # Start to make node feature file
    def init_node_features(self):
        save_path = os.path.join(self.search_path,'node_features')
        createFolder(save_path)

        if 'stacking' in self.problem:
            print("====[Stacking]====")
            ID_col = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Table']
            all_0 = [0]*len(ID_col)
               

            nf0 = pd.DataFrame({'ID': ID_col,'Type_Bowl': all_0, 'Type_Box': all_0, 'Type_Robot': all_0, \
                                            'Type_Table': all_0, 'Property_V': all_0, 'Property_G': all_0})
            nf0 = nf0.set_index("ID")

            for nf_inx in nf0.index:
                if "Box" in nf_inx:
                    nf0.loc[nf_inx, 'Type_Box'] = 1 
                    nf0.loc[nf_inx, 'Property_G'] = 1 
                elif "Bowl" in nf_inx:
                    nf0.loc[nf_inx, 'Type_Bowl'] = 1 
                    nf0.loc[nf_inx, 'Property_G'] = 1 
                elif "Robot" in nf_inx:
                    nf0.loc[nf_inx, 'Type_Robot'] = 1 
                elif "Table" in nf_inx:
                    nf0.loc[nf_inx, 'Type_Table'] = 1 
                else:
                    pass
                
            # Make 'Property_V' from self.example names
            for node in self.example.split('_'):
                if len(node) != 1:
                    for attached_box in node:
                        nf0.loc[f'Box{attached_box}', 'Property_V'] = 1 
            # print("nf0\n", nf0)
            
            if self.problem == 'stacking_v2':
                self.df_attach_v2(nf0)
                self.output_csv_file(0, 'node_features', nf0)
                print(nf0)
                print("====[stacking_v2]====")
            
            elif self.problem == 'stacking_v3':
                self.df_attach_v2(nf0)
                self.df_attach_v3(nf0)
                self.output_csv_file(0, 'node_features', nf0)
                print(nf0)
                print("====[stacking_v3]====")
                
            elif self.problem == 'stacking_v4':
                self.df_attach_v2(nf0)
                self.df_attach_v3(nf0)
                self.df_attach_v4(nf0)
                self.output_csv_file(0, 'node_features', nf0)
                print(nf0)
                print("====[stacking_v3]====")
            
            else:
                self.output_csv_file(0, 'node_features', nf0)
                print(nf0)
                print("====[stacking_5]====")
            
          

        elif 'mixing' in self.problem:
            print("====[Mixing]====")
            ID_col = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Bowl3','Bowl4','Bowl5','Bowl6','Bowl7', 'Table'] #0 ~ 13
            all_0 = [0]*len(ID_col)
               

            nf0 = pd.DataFrame({'ID': ID_col,'Type_Bowl': all_0, 'Type_Box': all_0, 'Type_Robot': all_0, \
                                            'Type_Table': all_0, 'Property_V': all_0, 'Property_G': all_0})
            nf0 = nf0.set_index("ID")

            for nf_inx in nf0.index:
                if "Box" in nf_inx:
                    nf0.loc[nf_inx, 'Type_Box'] = 1 
                    nf0.loc[nf_inx, 'Property_G'] = 1 
                elif "Bowl" in nf_inx:
                    nf0.loc[nf_inx, 'Type_Bowl'] = 1 
                    nf0.loc[nf_inx, 'Property_G'] = 1 
                elif "Robot" in nf_inx:
                    nf0.loc[nf_inx, 'Type_Robot'] = 1 
                elif "Table" in nf_inx:
                    nf0.loc[nf_inx, 'Type_Table'] = 1 
                else:
                    pass

            # Make 'Property_V' from self.example names    
            for node in self.example.split('_'):
                if len(node) != 1:
                    for attached_box in node:
                        nf0.loc[f'Box{attached_box}', 'Property_V'] = 1 
            # print("nf0\n", nf0)



            if self.problem == 'mixing_v2':
                self.df_attach_v2(nf0)
                self.output_csv_file(0, 'node_features', nf0)
                print(nf0)
                print("====[mixing_v2]====")
            
            elif self.problem == 'mixing_v3':
                self.df_attach_v2(nf0)
                self.df_attach_v3(nf0)
                self.output_csv_file(0, 'node_features', nf0)
                print(nf0)
                print("====[mixing_v3]====")
                
            elif self.problem == 'mixing_v4':
                self.df_attach_v2(nf0)
                self.df_attach_v3(nf0)
                self.df_attach_v4(nf0)
                self.output_csv_file(0, 'node_features', nf0)
                print(nf0)
                print("====[mixing_v3]====")
            
            else:
                self.output_csv_file(0, 'node_features', nf0)
                print(nf0)
                print("====[mixing_5]====")
        else:
            print("Check the problem")

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


###### Edge index ######
    # Relationship of the 'Box & Table (without attached boxes)', 'Bowl & Table'     
    def stack_edge_csv(self,ef0):
        for ef_inx in ef0.index:
            if "Box" in ef_inx and "+" not in ef_inx:
                ef0.loc[ef_inx, 'Table'] = 1 
                ef0.loc['Table', ef_inx] = 1  
            elif "Bowl" in ef_inx:
                ef0.loc[ef_inx, 'Table'] = 1 
                ef0.loc['Table', ef_inx] = 1  
            else:
                pass
        print("[Init edge index]\n",ef0)

    # Relationship of the 'Box & Table (without attached boxes)', 'Bowl & Table' , 'Box & Bowl(connecting with same numbers)'
    def mix_edge_csv(self,ef0):
        for ef_inx in ef0.index:
            if "Box" in ef_inx and "+" not in ef_inx:
                ef0.loc[ef_inx, f'Bowl{ef_inx[-1]}'] = 1
                ef0.loc[f'Bowl{ef_inx[-1]}', ef_inx] = 1
                # Table
                ef0.loc[ef_inx, 'Table'] = 1 
                ef0.loc['Table', ef_inx] = 1
            elif "Bowl" in ef_inx:
                ef0.loc[ef_inx, 'Table'] = 1 
                ef0.loc['Table', ef_inx] = 1  
            else:
                pass
        print("[Init edge index]\n",ef0)
    # Save problem as a dataframe of edge index
    def make_stacking_problem(self, list):
        nf = self.input_csv_file(0, 'node_features', None)
        node_index = nf['ID'].to_list()
        ef0 = pd.DataFrame(0, index= node_index, columns= list)
        ef0.index.name = "ID"
        self.stack_edge_csv(ef0)
        self.output_csv_file(0, 'edge_index', ef0)
    
    # Save problem as a dataframe of edge index
    def make_mixing_problem(self, list):
        nf = self.input_csv_file(0, 'node_features', None)
        node_index = nf['ID'].to_list()
        ef0 = pd.DataFrame(0, index= node_index, columns= list)
        ef0.index.name = "ID"
        self.mix_edge_csv(ef0)
        self.output_csv_file(0, 'edge_index', ef0)

    # Make initial edge index
    def init_edge_index(self):
        nf = self.input_csv_file(0, 'node_features', None)
        node_index = nf['ID'].to_list()

        if 'stacking' in self.problem:
            if self.problem == 'stacking_5':
                column_list = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Table']
                self.make_stacking_problem(column_list)

            elif self.problem == 'stacking_v2':
                column_list = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Table',\
                               f'{node_index[-1]}']
                self.make_stacking_problem(column_list)
                       
            elif self.problem == 'stacking_v3':
                column_list = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Table',\
                                f'{node_index[-2]}',f'{node_index[-1]}']
                self.make_stacking_problem(column_list)
                
   
             
            elif self.problem == 'stacking_v4':
                column_list = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Table',\
                                f'{node_index[-3]}',f'{node_index[-2]}',f'{node_index[-1]}']
                self.make_stacking_problem(column_list) 
            else:
                print("Wrong problem")
                
        elif 'mixing' in self.problem:
            if self.problem == 'mixing_5':            
                column_list = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Bowl3','Bowl4','Bowl5','Bowl6','Bowl7','Table']
                self.make_mixing_problem(column_list)
                
                
            elif self.problem == 'mixing_v2':
                column_list = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Bowl3','Bowl4','Bowl5','Bowl6','Bowl7','Table',\
                               f'{node_index[-1]}']
                self.make_mixing_problem(column_list)
            
            elif self.problem == 'mixing_v3':
                column_list = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Bowl3','Bowl4','Bowl5','Bowl6','Bowl7','Table',\
                                f'{node_index[-2]}',f'{node_index[-1]}']
                self.make_mixing_problem(column_list)
            
            elif self.problem == 'mixing_v4':
                column_list = ['Robot_hand','Box1','Box2','Box3','Box4','Box5','Bowl1','Bowl2','Bowl3','Bowl4','Bowl5','Bowl6','Bowl7','Table',\
                                f'{node_index[-3]}',f'{node_index[-2]}',f'{node_index[-1]}']
                self.make_mixing_problem(column_list) 
                       
            else:
                print("Wrong problem")
        

        
    # Pick edge index
    def pick_inx(self, n, obj1): # obj = ID number
        # Choose sample
        ei_csv = self.input_csv_file(n-1, 'edge_index', 0)
        pick_csv = ei_csv

        # Preconditions
        if obj1 != 'Robot_hand' and obj1 != 'Table': 
            if pick_csv.loc[obj1,'Robot_hand'] == 0 and pick_csv.loc['Robot_hand',obj1] == 0: # obj1 has no relationship with robot-hand

                # Remove 'on' relation (Table) 
                pick_csv.loc[obj1,'Table'] = 0
                pick_csv.loc['Table',obj1] = 0

                # Add 'in-grasp' relation (Robot-hand)
                pick_csv.loc[obj1,'Robot_hand'] = 1
                pick_csv.loc['Robot_hand',obj1] = 1
                print(f'\n[Pick[{str(obj1)}].csv] \n') 
                if "mixing" in self.problem:
                    # obj1 = Bowl4 -> Box4도 테이블 위에 X
                    pick_csv.loc['Table', f'Box{obj1[-1]}'] = 0
                    pick_csv.loc[f'Box{obj1[-1]}', 'Table'] = 0
                else:
                    pass

                # Save files
                self.output_csv_file(n, 'edge_index', pick_csv)
            
                return pick_csv
            
            else:
                print("\n----Check the '.csv' file again----\n")
        else:
            print("\n----Cannot pick this object----\n")


    # Place edge index
    def place_inx(self, n, obj1, obj2): 
        place_csv = self.input_csv_file(n-1, 'edge_index', 0)
        #### Simply attach object one by one 
        # Check obj1 and obj2 range
        if obj1 != 'Robot_hand' and obj1 != 'Table' and obj2 != 'Robot_hand':
            # 'in-grasp' relation (Robot hand O -> X) , object are not equal
            if place_csv.loc[obj1,'Robot_hand'] == 1 and place_csv.loc['Robot_hand',obj1] == 1:
                # Check obj1 and obj2 are equal
                if obj1 != obj2:
                    # Add 'on' relation with obj1 and obj2
                    place_csv.loc[obj1,obj2] = 1
                    place_csv.loc[obj2,obj1] = 1

                    # Remove 'in-grasp' relation (Robot hand)
                    place_csv.loc[obj1,'Robot_hand'] = 0
                    place_csv.loc['Robot_hand',obj1] = 0
                    
                    if 'stacking' in self.problem:
                        if self.problem != 'stacking_5':
                            nf = self.input_csv_file(n, 'node_features', None)
                            attached_boxes = nf[nf['Type_Attached_Boxes'] == 1]["ID"].to_list()
                            
                            for attached_box in attached_boxes:
                                if str(attached_box) == str(obj1) + str(' + ') +str(obj2):
                                    # attached_boxes끼리 모으기
                                    print("\n[Attached]",attached_box)

                                    place_csv.loc[attached_box, :] = 0
                                    place_csv.loc[:, f'{attached_box}'] = 0               
                                    place_csv.loc[attached_box, 'Table'] = 1
                                    place_csv.loc['Table', f'{attached_box}'] = 1

                                    # attached_boxes와 object 사이 관계 hierarchical하게  [Place_[Box3]_on_[Box4 + Box5].csv] 
                                    # place_csv.loc[obj1,f'{attached_box}'] = 1
                                    # place_csv.loc[attached_box, obj1] = 1
                                    # place_csv.loc[obj2, f'{attached_box}'] = 1
                                    # place_csv.loc[attached_box, obj2] = 1
                                    
                                    # split_attached_boxes와 object 사이 관계성 
                                    # split_attached_boxes = attached_box.split(" + ")
                                    # print("\n[Split attached boxes]", split_attached_boxes)
                                    # print(f"\n=={obj1} and {obj2} is attached==")
                            else:
                                pass
                        else:
                            pass
                    

                    
                    print(f'\n[Place_[{str(obj1)}]_on_[{str(obj2)}].csv] \n') 

                    ### Save files
                    self.output_csv_file(n, 'edge_index', place_csv)

                    return place_csv
                
                else:
                    print("----Object1 and object2 are equal----")
            else:
                pass
            #     print("\n----Robot hand does not hold obj1. Please check the '.csv' file again----\n")
        else:
            print("----Cannot place this object----")
            
            

    # def attach_inx(self, n, obj1, obj2):
    #     attach_csv = self.input_csv_file(n-1, 'edge_index', 0)
    #     #### Simply attach object one by one 
    #     nf = self.input_csv_file(n, 'node_features', None)

    #     attached_boxes = nf[nf['Type_Attached_Boxes'] == 1]["ID"].to_list()[0]

        
    #     if obj1 != 0 and obj1 != 8 and obj2 != 0 and obj2 != 8:
    #         if attach_csv.loc[obj1,f'{obj2}'] == 1 and attach_csv.loc[obj2,f'{obj1}'] == 1:
    #             # attach_csv = attach_csv.drop(index= obj1)
    #             # attach_csv = attach_csv.drop(columns= f'{obj1}')
    #             # attach_csv = attach_csv.drop(index= obj2)
    #             # attach_csv = attach_csv.drop(columns= f'{obj2}')

                
    #             # attach box에 대한 행과 열 추가
    #             # attach_csv.loc[attached_boxes] = 0
    #             # attach_csv[f'{attached_boxes}'] = 0
                
    #             # 8번 Table 위에 올려두기
    #             attach_csv.loc[attached_boxes, '8'] = 1
    #             attach_csv.loc[8, f'{attached_boxes}'] = 1

    #             ## Save files
    #             self.output_csv_file(n, 'edge_index', attach_csv)
     
    #             return attach_csv
        
    #         else:
    #             print("----Obj1 is not <on> Obj2----")
    #     else:
    #         print("----Cannot attach those object----")
        

    # Pour edge index
    def pour_inx(self,n, obj1, obj2):
        pour_csv = self.input_csv_file(n-1, 'edge_index', 0)
        pour_objects = pour_csv.index
        # print('Pour object', pour_objects)
        # Node feature 따라서 만들고 objects은 bowl이어야 함
        if 'Bowl' in obj1 and 'Bowl' in obj2:
            for pour_obj in pour_objects:
                if pour_obj != 'Robot_hand' and pour_obj != 'Table':
                    if pour_csv.at[obj1, pour_obj] == 1 and pour_csv.at[pour_obj, obj1] == 1:
                        print("[[Pour obj]]", pour_obj)
                        pour_csv.at[obj1, pour_obj] = 0
                        pour_csv.at[pour_obj, obj1] = 0
                        pour_csv.at[obj2, pour_obj] = 1
                        pour_csv.at[pour_obj, obj2] = 1
                else:
                    pass
                  
            print(f'\n[Pour_[{str(obj1)}]_on_[{str(obj2)}].csv] \n') 
            ### Save files
            self.output_csv_file(n, 'edge_index', pour_csv)
            return pour_csv
    
        else:
            print("----Object is not a bowl----")

    # Mix edge index
    def mix_inx(self,n, obj1, obj2):
        nf = self.input_csv_file(n, 'node_features', None)
        attached_boxes = nf[nf['Type_Attached_Boxes'] == 1]["ID"].to_list()
        
        mix_csv = self.input_csv_file(n-1, 'edge_index', 0)
        # print('Pour object', pour_objects)
        # Node feature 따라서 만들고 objects은 bowl이어야 함
        if 'Box' in obj1 and 'Box' in obj2:
            for mixed_obj in attached_boxes:
                if str(obj1) + str(' + ') +str(obj2) == str(mixed_obj):
                    if mixed_obj != 'Robot_hand' and mixed_obj != 'Table':
                        print("[[Mixed obj]]", mixed_obj)
                        mix_csv.loc[obj1, obj2] = 1
                        mix_csv.loc[obj2, obj1] = 1
                        mix_csv.loc[obj1, attached_boxes] = 1
                        mix_csv.loc[attached_boxes, obj1] = 1
                        mix_csv.loc[obj2, attached_boxes] = 1
                        mix_csv.loc[attached_boxes, obj2] = 1
                        mix_csv.loc[attached_boxes, 'Bowl6'] = 1
                        mix_csv.loc['Bowl6',attached_boxes] = 1
                       
                else:
                    pass
                  
            print(f'\n[Mix_[{str(obj1)}]_and_[{str(obj2)}].csv] \n') 
            ### Save files
            self.output_csv_file(n, 'edge_index', mix_csv)
            return mix_csv
    
        else:
            print("----Object is not a box----")


    

###### Edge attr ######
    # Make initial edge attribute file
    def init_edge_attr(self,n,num):
        list_attr = []
        list_attr1 = []

        # Dataframe
        nf = self.input_csv_file(n, 'node_features', None)
        
        for i in range(num):   
            ef = self.input_csv_file(i, 'edge_index', 0)
            ID_list = list(map(str, ef.columns))
            for index in range(len(ID_list)):
                for column in range(len(ID_list)):
                    if ef.iat[index, column] == 1:   
                        list_attr1.append((ID_list[index], ID_list[column]))
        
        # 자체 중복 방지
        for v in list_attr1:
            if v not in list_attr:
                list_attr.append(v)
        # print('list_attr', list_attr, len(list_attr))
       


        list_0 = [0 for i in range(len(list_attr))]
        

        # print(list_attr, "length", len(list_attr))
        edge_attr0_csv = pd.DataFrame({'ID': list_attr, 'rel_on_right':list_0, 'rel_on_left': list_0, \
                                        'rel_in_right':list_0, 'rel_in_left': list_0, 
                                        'rel_in_grasp':list_0, 'rel_grasp': list_0, 'rel_attach': list_0 })
                                        #'pos_x': list_0, 'pos_y': list_0, 'pos_z': list_0, \
                                        #'pos_roll':list_0, 'pos_pitch':list_0, 'pos_yaw':list_0})
        edge_attr0_csv = edge_attr0_csv.set_index("ID")
        

        # print(list_attr1)
        for node_pair in list_attr1:
            np0 = node_pair[0]
            np1 = node_pair[1]
            if 'stacking' in self.problem:
                if self.problem != 'stacking_5':
                    # attached_boxes = nf[nf['Type_Attached_Boxes'] == 1]["ID"].to_list()
                    # for attached_box in attached_boxes:     
                    #     # print("[Attach boxes]", attached_box)   
                    if np0 == 'Table':
                        if "+" in np1:
                            print("np1", node_pair)
                            # print(type(attached_box), type(np1))
                            edge_attr0_csv.loc[[node_pair], :] = 0
                        else:   
                            print("else",node_pair)
                            edge_attr0_csv.loc[[node_pair], 'rel_on_left'] = 1
                    if np1 == 'Table':
                        if "+" in np0:
                            edge_attr0_csv.loc[[node_pair], :] = 0
                        else:
                            edge_attr0_csv.loc[[node_pair], 'rel_on_right'] = 1 
                else:
                    if np0 == 'Table':
                        edge_attr0_csv.loc[[node_pair], 'rel_on_left'] = 1
                    if np1 == 'Table':
                        edge_attr0_csv.loc[[node_pair], 'rel_on_right'] = 1
                    
            elif 'mixing' in self.problem:
                if 'Box' in np0 and 'Bowl' in np1 and np0[-1] == np1[-1]:
                    edge_attr0_csv.loc[[node_pair], 'rel_in_right'] = 1
                if 'Bowl' in np0 and 'Box' in np1 and np0[-1] == np1[-1]:
                    edge_attr0_csv.loc[[node_pair], 'rel_in_left'] = 1
                if np0 == 'Table':
                    edge_attr0_csv.loc[[node_pair], 'rel_on_left'] = 1
                if np1 == 'Table':
                    edge_attr0_csv.loc[[node_pair], 'rel_on_right'] = 1
            else:
                print("Wrong problem")


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
            if np0 == obj1 and np1 == 'Table':
                edge_attr_csv.loc[[node_pair], :] = 0
                if 'mixing' in self.problem: # Bowl4 안에 Box4도 테이블 위에 X
                    print("u", np0, np1)
                    edge_attr_csv.loc[[f"('Box{np0[-1]}', '{np1}')"], :] = 0
                    # [[f"('{np0}', '{obj2}')"], 'rel_in_right'] = 1
            if np1 == obj1 and np0 == 'Table':
                edge_attr_csv.loc[[node_pair], :] = 0
                if 'mixing' in self.problem:
                    print("u", np0, np1)
                    edge_attr_csv.loc[[f"('{np0}', 'Box{np1[-1]}')"], :] = 0
            # Grasp 'obj1 in robot-hand
            if np0 == 'Robot_hand' and np1 == obj1:
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[node_pair], 'rel_grasp'] = 1
            if np1 == 'Robot_hand' and np0 == obj1:
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[node_pair], 'rel_in_grasp'] = 1 
            
    
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
            
            # if 'stacking' in self.problem:
            # Delete previous relationship
            edge_attr_csv.loc[[node_pair], 'rel_grasp'] = 0
            edge_attr_csv.loc[[node_pair], 'rel_in_grasp'] = 0
            # Obj1 is on-relationship with Obj2
            if np0 == obj1 and np1 == obj2:
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[node_pair], 'rel_on_right'] = 1
            if np0 == obj2 and np1 == obj1:
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[node_pair], 'rel_on_left'] = 1   
                    
                ### normal이 아닐 때        # If there is an Property_V:
                if self.problem in 'stacking'
                if self.problem != 'stacking_5' and self.problem != 'mixing_5':    
                    nf = self.input_csv_file(n, 'node_features', None)             
                    attached_boxes = nf[nf['Type_Attached_Boxes'] == 1]["ID"].to_list()
                    for attached_box in attached_boxes:
                        if str(attached_box) == str(obj1) + str(' + ') +str(obj2):
                            print("TT", attached_box)
                            if np0 == obj1 and np1 == obj2:
                                edge_attr_csv.loc[[node_pair], :] = 0
                                edge_attr_csv.loc[[node_pair], 'rel_attach'] = 1
                            if np0 == obj2 and np1 == obj1:
                                edge_attr_csv.loc[[node_pair], :] = 0
                                edge_attr_csv.loc[[node_pair], 'rel_attach'] = 1   
                            if np0 == attached_box and np1 == 'Table':
                                edge_attr_csv.loc[[node_pair], :] = 0
                                edge_attr_csv.loc[[node_pair], 'rel_on_right'] = 1                    
                            if np1 == attached_box and np0 == 'Table':
                                edge_attr_csv.loc[[node_pair], :] = 0
                                edge_attr_csv.loc[[node_pair], 'rel_on_left'] = 1 
                            # hierarchical
                            # if np0 == attached_box and np1 == obj1:
                            #     edge_attr_csv.loc[[node_pair], :] = 0
                            #     edge_attr_csv.loc[[node_pair], 'rel_hierarchical_left'] = 1 
                            # if np1 == attached_box and np0 == obj1:
                            #     edge_attr_csv.loc[[node_pair], :] = 0
                            #     edge_attr_csv.loc[[node_pair], 'rel_hierarchical_right'] = 1 
                            # if np0 == attached_box and np1 == obj1:
                            #     edge_attr_csv.loc[[node_pair], :] = 0
                            #     edge_attr_csv.loc[[node_pair], 'rel_hierarchical_left'] = 1 
                            # if np1 == attached_box and np0 == obj1:
                            #     edge_attr_csv.loc[[node_pair], :] = 0
                            #     edge_attr_csv.loc[[node_pair], 'rel_hierarchical_right'] = 1 
                            
                    
                        else:
                            pass
                    else:
                        pass
            elif 'mixing' in self.problem:
                pass

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
            if np0 == obj1 and np1 != 'Robot_hand' and np1 != 'Table':
                # print(node_pair)
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[f"('{obj2}', '{np1}')"], 'rel_in_left'] = 1
                # edge_attr_csv.loc[[f"('{np1}', '{obj2}')"], 'rel_in_left'] = 5
                

            if np1 == obj1 and np0 != 'Robot_hand' and np0 != 'Table' :
                # print(node_pair)
                edge_attr_csv.loc[[node_pair], :] = 0
                edge_attr_csv.loc[[f"('{np0}', '{obj2}')"], 'rel_in_right'] = 1
        print(f"\n[Pour[{obj1}]_to_[{obj2}].csv]\n",edge_attr_csv)

        # SAVE PATH (edge_attr)
        self.output_csv_file(n, 'edge_attr', edge_attr_csv)
        print(f"\n----{self.example}_ea{n}.csv is saved----")

    # Mix edge attribute
    def mix_attr(self, n, obj1, obj2):
        nf = self.input_csv_file(0,'node_features',0)
        # Property velcro, attach boxes
        attach_v = nf[nf['Property_V'] == 1].index.to_list()
        attached_boxes = nf[nf['Type_Attached_Boxes'] == 1].index.to_list()
        # Edge attr file
        edge_attr_csv = self.input_csv_file(n-1,'edge_attr',0)
        edge_attr_index = edge_attr_csv.index.to_list()
        new_list = self.list_changer(edge_attr_index)
        
      
        # print('Pour object', pour_objects)
        # Node feature 따라서 만들고 objects은 bowl이어야 함
        for npr in new_list:           
            np0 = npr[0]
            np1 = npr[1]
            npa = (np0, np1)
            node_pair= "('{}', '{}')".format(*npa)
        
    
            if obj1 in attach_v and obj2 in attach_v:
                for attached_obj in attached_boxes:
                    if str(obj1) + str(' + ') +str(obj2) == str(attached_obj):
                        if np0 == obj1 and np1 == obj2:
                            # print("T", node_pair)
                            # obj1-obj2 and obj2-obj1 attach
                            edge_attr_csv.loc[[node_pair], 'rel_attach'] = 1
                        if np1 == obj1 and np0 == obj2:
                            edge_attr_csv.loc[[node_pair], 'rel_attach'] = 1
                        
                  ############################## 지금은 Bowl6에 한정, Bowl번호에 상관없이 mix할 수 있도록 수정 ###########################
                        if np0 == attached_obj and np1 == 'Bowl6' :
                            edge_attr_csv.loc[[node_pair], 'rel_in_right'] = 1
                        if np1 == attached_obj and np0 == 'Bowl6' :
                            edge_attr_csv.loc[[node_pair], 'rel_in_left'] = 1
                                
                            # Hierarchy 
                            # if edge_attr_csv.at[[f"('{np0}', '{mix_bowl}')"], 'rel_on_right'] == 1:
                            #     edge_attr_csv.loc[[f"('{attached_obj}', '{mix_bowl}')"], 'rel_attach'] = 4
                            # if edge_attr_csv.at[[f"('{np1}', '{mix_bowl}')"], 'rel_on_right'] == 1:
                            #     edge_attr_csv.loc[[f"('{attached_obj}', '{mix_bowl}')"], 'rel_attach'] = 5
                            # edge_attr_csv.loc[[f"('{attached_obj}', '{np0}')"], 'rel_hierarcy'] = 4
                            # edge_attr_csv.loc[[f"('{attached_obj}', '{np1}')"], 'rel_hierarcy'] = 4
                        
                            # edge_attr_csv.loc[[f"('{np0}', '{attached_obj}')"], 'rel_hierarcy'] = 5
                            # edge_attr_csv.loc[[f"('{np1}', '{attached_obj}')"], 'rel_hierarcy'] = 5
                        # print("Nodepair", node_pair)
                        
                        # edge_attr_csv.loc[[f"('{obj1}', '{obj2}')"], 'rel_in_left'] = 1

        print(f"\n[Mix[{obj1}]_and_[{obj2}].csv]\n",edge_attr_csv)

        # SAVE PATH (edge_attr)
        self.output_csv_file(n, 'edge_attr', edge_attr_csv)
        print(f"\n----{self.example}_ea{n}.csv is saved----")



    ############################## Make graph ##################################
    # 아이콘이 들어간 graph -> 시각화 자료 사용할 때 씀 (attached된 bow도 사진을 추가해야하는 상태)
    # def make_graph(self, fig_num, pos):
        
    #     # Weight 부여되면 굵어지게
    #     list_edge_attr = []
    #     list_edge_on = []
    #     list_edge_grasp = []
    #     # list_node_pair = []

    #     # nf_csv = self.input_csv_file(fig_num, 'node_features', 0)
    #     # nf_index = nf_csv.index.to_list()
    #     # # print("[Nf index]", nf_index)

    #     # Connect edge
    #     edge_attr_csv = self.input_csv_file(fig_num, 'edge_attr', 0)
    #     ea_index = edge_attr_csv.index.to_list()
    #     ea_inx = self.list_changer(ea_index)
    #     print("[EA index]", ea_inx)       

    #     # edge_attr의 column 데이터 list로 가져오기
    #     col = edge_attr_csv.columns.to_list()
    #     # edge_attr file에서 'rel'이 들어간 문자열 정보 가져오기 
    #     ea_col = [col[i] for i in range(len(col)) if col[i].find('rel') == 0]    

    #     # print("\n[ea col]",ea_col)
     
    #     for npr in ea_inx:           
    #         np0 = npr[0]
    #         np1 = npr[1]
    #         npa = (np0, np1)
    #         node_pair= "('{}', '{}')".format(*npa)
    #         for j in range(len(ea_col)):
    #             if edge_attr_csv.at[node_pair, ea_col[j]] == 1:
    #                 if ea_col[j] == 'rel_on_right':
    #                     attr = ea_col[j].replace('rel_on_right', 'On')
    #                     list_edge_on.append(attr)
    #                 elif ea_col[j] == 'rel_on_left':
    #                     attr = ea_col[j].replace('rel_on_left', 'On')
    #                     list_edge_on.append(attr)
    #                 elif ea_col[j] == 'rel_in_right':
    #                     attr = ea_col[j].replace('rel_in_right', 'In')
    #                     list_edge_on.append(attr)
    #                 elif ea_col[j] == 'rel_in_left':
    #                     attr = ea_col[j].replace('rel_in_left', 'In')
    #                     list_edge_on.append(attr)
    #                 elif ea_col[j] == 'rel_in_grasp':
    #                     attr = ea_col[j].replace('rel_in_grasp', 'Grasp')
    #                     list_edge_grasp.append(attr)
    #                 elif ea_col[j] == 'rel_grasp':
    #                     attr = ea_col[j].replace('rel_grasp','Grasp')
    #                     list_edge_grasp.append(attr)
    #                 elif ea_col[j] == 'rel_attach':
    #                     attr = ea_col[j].replace('rel_attach','Attach')
    #                     list_edge_on.append(attr)
    #                 elif ea_col[j] == 'rel_hierarchical_right':
    #                     attr = ea_col[j].replace('rel_hierarchical_right','Hierarchy')
    #                     list_edge_on.append(attr)
    #                 elif ea_col[j] == 'rel_hierarchical_left':
    #                     attr = ea_col[j].replace('rel_hierarchical_left','Hierarchy')
    #                     list_edge_on.append(attr)
    #                 else:
    #                     print("----Re-check relations----")
    #                 list_edge_attr.append(attr)
    
        
    #     print("\n[List edge attribute]:",list_edge_attr)
     

    #     ################### Make graph ####################
    #     import matplotlib.pyplot as plt
    #     import networkx as nx
    #     import PIL
        
    #     icon_path = os.path.join(self.FILEPATH, 'icons')
        
    #     # Image URLs for graph nodes
    #     icons = {
    #         "Robot0": f"{self.FILEPATH}/icons/robot_hand.jpeg",
    #         "Block1": f"{self.FILEPATH}/icons/block1.jpg",
    #         "Block2": f"{self.FILEPATH}/icons/block2.jpg",
    #         "Block3": f"{self.FILEPATH}/icons/block3.jpg",
    #         "Block4": f"{self.FILEPATH}/icons/block4.jpg",
    #         "Block5": f"{self.FILEPATH}/icons/block5.jpg",
    #         "Bowl6": f"{self.FILEPATH}/icons/bowl6.jpeg",
    #         "Bowl7": f"{self.FILEPATH}/icons/bowl7.webp",
    #         "Table": f"{self.FILEPATH}/icons/table_icon.jpg",
    #     }

    #     # Load images
    #     images = {k: PIL.Image.open(fname) for k, fname in icons.items()}
        
        
    #     # Generate graph
    #     g = nx.Graph()
        
    
    #     # Add nodes
    #     # g.add_nodes_from(nodes, images = images["Block1"])
    #     g.add_node("Robot_hand", images = images["Robot0"])
    #     g.add_node("Box1", images = images["Block1"])
    #     g.add_node("Box2", images = images["Block2"])
    #     g.add_node("Box3", images = images["Block3"])
    #     g.add_node("Box4", images = images["Block4"])
    #     g.add_node("Box5", images = images["Block5"])
    #     g.add_node("Box6", images = images["Bowl6"])
    #     g.add_node("Box7", images = images["Bowl7"])
    #     g.add_node("Box8", images = images["Table"])
        
        
       
    #     # Add edges
    #     for i in range(len(list_edge_attr)):
    #         for node_pair in ea_inx:
    #             np0 = node_pair[0]
    #             np1 = node_pair[1]
    #             edges = [(np0, np1, {'label': list_edge_attr[i]})]
    #             g.add_edges_from(edges)
    #     #     g.add_edges_from(ea_inx[i], label = f'{list_edge_attr[i]}')
               
    #     # edge_labels = nx.get_edge_attributes(g,'label')
    #     # print("\n[Edge labels]:",edge_labels)
      

    #     # POS 1 사진으로 node image 가져오는 것 가능
        
    #     # pos 지정 => x,y 좌표 array값으로 받아서 사용할 수 있음
    #     # manually specify node position
    #     # pos = nx.spring_layout(g)
    #     # pos = nx.shell_layout(g)
        
    #     # check the position

    #     # Get a repreducible layout and create figure
    #     fig, ax = plt.subplots() 
        
    #     # Transform from data coordinates
    #     tr_figure = ax.transData.transform
    #     # Transform from display to figure coordinates
    #     tr_axes = fig.transFigure.inverted().transform

       
         
    #     # Select the size of the image
    #     icon_size =  0.065 #(ax.get_xlim()[1] - ax.get_xlim()[0])*0.08 # 0.08
    #     icon_center = icon_size / 2.0                          # 0.025
      
        
    #     # Show title
    #     title_font = {'fontsize':14, 'fontweight':'bold'}
    #     plt.title("Present state", fontdict = title_font)  
        
        

    #     egrasp = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "Grasp"]
    #     eon = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "On"]
    #     ein = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "In"]
    #     eattach = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "Attach"]

    #     # print(egrasp)
    #     # print(eon)
    #     # print(ein)
    #     # print(eattach)
        
    #     # Draw edges from edge attributes
    #     # styles = ['filled', 'rounded', 'rounded, filled', 'dashed', 'dotted, bold']
    #     nx.draw_networkx_edges(G=g, pos=pos, edgelist=egrasp, width=6, alpha=0.5, edge_color="b", style= "dotted")
    #     nx.draw_networkx_edges(G=g, pos=pos, edgelist=eon, width=3, alpha=0.5, edge_color="black")
    #     nx.draw_networkx_edges(G=g, pos=pos, edgelist=ein, width=4, alpha=0.5, edge_color="r")
    #     nx.draw_networkx_edges(G=g, pos=pos, edgelist=eattach, width=4, alpha=0.5, edge_color="r")

    #     # Draw edge labels from edge attributes
    #     # nx.draw_networkx_edge_labels(G= g, pos = pos, ax=ax, edge_labels = edge_labels, font_size = 10)
      
    #     # print("[G nodes]", g.nodes)
                
    #     for i,n in enumerate(g.nodes):
    #         # print("[i]",i,n)
    #         # print("[pos]",pos)
    #         xf, yf = tr_figure(pos[i])
    #         xa, ya = tr_axes((xf, yf))

    #         # get overlapped axes and plot icon
    #         a = plt.axes([xa-icon_center, ya-icon_center , icon_size, icon_size])
    #         a.set_aspect('equal')
    #         a.imshow(g.nodes[n]['images']) # print(g.nodes[n]) #dictionary에 'image' -> 'images'로 변경됨
    #         a.axis("off")
             
        
    #     # plt.figure(figsize=(10,8))  
    #     ### Check the graphs
    #     plt.show() # 

    #     ### Save the graph files
    #     # nx.draw(g) # 저장할 때
    #     # graph_path = os.path.join(self.FILEPATH, self.problem, 'graph_image')
    #     # createFolder(graph_path)
    #     # task_name = 'task' + str(fig_num) + '.png'
    #     # save_graph_path = os.path.join(graph_path, task_name)
    #     # plt.savefig(save_graph_path)



    # Graph로 node_feature, edge_index, edge_attr 확인하는 용도
    def check_graph(self, fig_num, node1, node2):
    
        # Node list
        nf_csv = self.input_csv_file(fig_num, 'node_features', 0)
        nf_index = nf_csv.index.to_list()
       
        
        # Connect edge
        edge_attr_csv = self.input_csv_file(fig_num, 'edge_attr', 0)
        ea_index = edge_attr_csv.index.to_list()
        ea_inx = self.list_changer(ea_index)
            

        # edge_attr의 column 데이터 list로 가져오기
        col = edge_attr_csv.columns.to_list()
        # edge_attr file에서 'rel'이 들어간 문자열 정보 가져오기 
        ea_col = [col[i] for i in range(len(col)) if col[i].find('rel') == 0]    

        # print("\n[ea col]",ea_col)
     
        import matplotlib.pyplot as plt
        import networkx as nx
 
        
        
        # Generate graph
        g = nx.DiGraph() # 방향성
        # g = nx.Graph() # 무방향성
        
        g.add_nodes_from(nf_index)


        for npr in ea_inx:           
            np0 = npr[0]
            np1 = npr[1]
            npa = (np0, np1)
            node_pair= "('{}', '{}')".format(*npa)
            for rel in ea_col:
                if edge_attr_csv.at[node_pair, rel] == 1:
                    if rel == 'rel_on_right' or rel == 'rel_on_left':
                        new_rel = rel.replace(rel, 'On')
                    elif rel == 'rel_in_grasp' or rel == 'rel_grasp':
                        new_rel = rel.replace(rel, 'Grasp')
                    elif rel == 'rel_in_right' or rel == 'rel_in_left':
                        new_rel = rel.replace(rel, 'In')
                    elif rel == 'rel_attach':
                        new_rel = rel.replace(rel, 'Attach')
                    # elif rel == 'rel_hierarchical_right' or rel == 'rel_hierarchical_left':
                    #     new_rel = rel.replace(rel, 'Hierarchy')

                    networkx_edges = [(np0, np1, {'label':new_rel})]
                    g.add_edges_from(networkx_edges)

                  


        ################### Make graph ####################

        # Can manually specify node position
        # pos = nx.spring_layout(g) # 분산된 형태
        pos = nx.shell_layout(g) # 둥근 형태
       
        # # check the position

        
        # Show title
        plt.figure(figsize=(10,8))
        title_font = {'fontsize':14, 'fontweight':'bold'}
        plt.title(f"Task{fig_num}", fontdict = title_font)  
        
        print("[Graph info]", g)

        egrasp = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "Grasp"]
        eon = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "On"]
        ein = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "In"]
        eattach = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "Attach"]
        # ehierarchy = [(u, v) for (u, v, d) in g.edges(data=True) if d["label"] == "Hierarchy"]

        print("[Grasp]", egrasp)
        print("[On]", eon)
        print("[In]", ein)
        print("[Attach]",eattach)
        


        
        # Draw edges from edge attributes
        
        # Task에 해당되는 node의 색깔 다르게 
        val_map = {node1: 0.5, node2: 0.5}
        values = [val_map.get(node, 0.25) for node in g.nodes()]
        nx.draw_networkx_nodes(G=g, pos=pos, cmap= plt.get_cmap('rainbow'), node_color= values, node_size=400)

        
        nx.draw_networkx_labels(G=g, pos=pos, labels= {node: node for node in g.nodes()})
        nx.draw_networkx_edges(G=g, pos=pos, edgelist=egrasp, width=3, alpha=0.5, edge_color='blue', style= "dotted")
        nx.draw_networkx_edges(G=g, pos=pos, edgelist=eon, width=3, alpha=0.5, edge_color= 'black')
        nx.draw_networkx_edges(G=g, pos=pos, edgelist=ein, width=3, alpha=0.5, edge_color="r")
        nx.draw_networkx_edges(G=g, pos=pos, edgelist=eattach, width=3, alpha=0.5, edge_color="b")

        lgrasp = {pairs: 'Grasp' for pairs in egrasp}
        lon = {pairs: 'On' for pairs in eon}
        lin = {pairs: 'In' for pairs in ein}
        lattach = {pairs: 'Attach' for pairs in eattach}
        
        # # Draw edge labels from edge attributes
        
        nx.draw_networkx_edge_labels(G= g, pos = pos, edge_labels = lgrasp, font_size = 10)
        nx.draw_networkx_edge_labels(G= g, pos = pos, edge_labels = lon, font_size = 10)
        nx.draw_networkx_edge_labels(G= g, pos = pos, edge_labels = lin, font_size = 10)
        nx.draw_networkx_edge_labels(G= g, pos = pos, edge_labels = lattach, font_size = 10)
        
        plt.show()
     
        


###################### Checking from networkx ############
    # draw_networkx
    # draw_networkx_nodes
    # draw_networkx_edges
    # draw_networkx_labels
    # draw_networkx_edge_labels



###################################### Graph에서 구체적인 position을 정할 때 사용 ##################################################


stack_pos0 = {
    0: [0.33, 0.35],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.67, 0.28],
    4: [0.60, 0.35],
    5: [0.5, 0.38],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

stack_pos1 = {
    0: [0.60, 0.5],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.67, 0.28],
    4: [0.60, 0.35],
    5: [0.5, 0.38],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

stack_pos2 = {
    0: [0.65, 0.45],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.6, 0.35],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

stack_pos3 = {
    0: [0.6, 0.5],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.6, 0.35],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}


stack_pos4 = {
    0: [0.6, 0.5],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.5, 0.7],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

stack_pos5 = {
    0: [0.4, 0.6],
    1: [0.33, 0.28],
    2: [0.40, 0.35],
    3: [0.5, 0.7],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}


stack_pos6 = {
    0: [0.4, 0.6],
    1: [0.35, 0.3],
    2: [0.5, 0.9],
    3: [0.5, 0.7],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}


stack_pos7 = {
    0: [0.35, 0.6],
    1: [0.35, 0.3],
    2: [0.5, 0.9],
    3: [0.5, 0.7],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}


stack_pos8 = {
    0: [0.35, 0.5],
    1: [0.5, 1.1],
    2: [0.5, 0.9],
    3: [0.5, 0.7],
    4: [0.5, 0.5],
    5: [0.5, 0.3],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

stack_pos = [stack_pos0, stack_pos1, stack_pos2, stack_pos3, stack_pos4, stack_pos5, stack_pos6, stack_pos7, stack_pos8]


mix_pos0 = {     
    0: [0.33, 0.35],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.40, 0.35],
    5: [0.33, 0.28],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

mix_pos1 = {     
    0: [0.33, 0.4],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.40, 0.35],
    5: [0.33, 0.28],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

mix_pos2 = {     
    0: [0.33, 0.35],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.40, 0.35],
    5: [0.21, 0.23],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

mix_pos3 = {     
    0: [0.4, 0.47],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.4, 0.35],
    5: [0.18, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}

mix_pos4 = {     
    0: [0.4, 0.47],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.21, 0.25],
    5: [0.18, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos5 = {     
    0: [0.5, 0.5],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.5, 0.38],
    4: [0.21, 0.25],
    5: [0.18, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos6 = {     
    0: [0.5, 0.5],
    1: [0.67, 0.28],
    2: [0.60, 0.35],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos7 = {     
    0: [0.6, 0.47],
    1: [0.67, 0.28],
    2: [0.6, 0.35],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos8 = {     
    0: [0.55, 0.28],
    1: [0.67, 0.28],
    2: [0.43, 0.27],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos9 = {     
    0: [0.67, 0.4],
    1: [0.67, 0.28],
    2: [0.43, 0.27],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos10 = {   
    0: [0.67, 0.4],
    1: [0.5, 0.22],
    2: [0.43, 0.27],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos11 = {   
    0: [0.4, 0.31],
    1: [0.5, 0.22],
    2: [0.43, 0.27],
    3: [0.3, 0.3],
    4: [0.17, 0.27],
    5: [0.1, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos12 =  {   
    0: [0.4, 0.3],
    1: [0.9, 0.22],
    2: [0.83, 0.27],
    3: [0.7, 0.3],
    4: [0.57, 0.27],
    5: [0.5, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}
mix_pos13 =  {   
    0: [0.4, 0.3],
    1: [0.9, 0.22],
    2: [0.83, 0.27],
    3: [0.7, 0.3],
    4: [0.57, 0.27],
    5: [0.5, 0.22],
    6: [0.3, 0.2],
    7: [0.7, 0.2],
    8: [0.5, 0.1]
}


mix_pos = [mix_pos0, mix_pos1,mix_pos2,mix_pos3,mix_pos4,mix_pos5,mix_pos6,mix_pos7,mix_pos8,mix_pos9,mix_pos10,mix_pos11, mix_pos12, mix_pos13]