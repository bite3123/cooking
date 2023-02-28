from seq_demo import *




if __name__ == '__main__':
    # make_data = MakeDataset(problem = 'stacking_5', example = 'ex_1_2_3_4_5')
    

    ## Saving files
    # FILEPATH, _ = os.path.split(os.path.realpath(__file__))
    # print(FILEPATH)
    # save_path = os.path.join(FILEPATH,'stacking_velcro2',file_name, 'node_features')
    # save_path = os.path.join(FILEPATH,'stacking_velcro2',file_name, 'edge_index')
    # save_path = os.path.join(FILEPATH,'stacking_velcro2',file_name, 'edge_attr')
    # createFolder(save_path)
  

# Pick, place, pour 

    # print(make_data.pick(file_num = 0, obj1 = 5))
    # print(make_data.place(file_num = 0, obj1 = 5, obj2= 6))
    # print(make_data.pick(file_num = 0, obj1 = 4))
    # print(make_data.place(file_num = 0, obj1 = 4, obj2= 6))
    # print(make_data.pick(file_num = 0, obj1 = 3))
    # print(make_data.place(file_num = 4, obj1 = 3, obj2= 6))
   
    # print(make_data.pick(file_num = 0, obj1 = 5))
    # print(make_data.place(file_num = 0, obj1 = 5, obj2= 6))
    # print(make_data.pick(file_num = 1, obj1 = 6))
    # print(make_data.pour(file_num = 0, obj1 = 6 , obj2 = 7))


    


    make_data = MakeDataset(problem = 'mixing_5', example = 'mix_ex_1_2_3_4_5')
    # make_data = MakeDataset(problem = 'stacking_5', example = 'ex_1_2_3_4_5')
  
    for a in range(0, 12): # stacking (0,9) 0~8, mixing (0,12) 0~11
    ################### Call sample data ####################
        # action_seq = ['pick','place','pick','place','pick','place','pick','place','pick','place']
        print(f"\n[[[[[Task{a}]]]]]")
        print(make_data.sample_data(i=a))
        #[Warning]#### 0으로 시작 # print(make_data.init_edge_attr(file_num = a))


        ### Checking graphs
        # make_data.make_graph(fig_num=a, pos = mix_pos[a])
        # make_data.make_edge_index(i=a) 
        # make_data.make_edge_attr(i=a) # def make_edge_index실행 후에 돌려

        # # Object 5
        # print(make_data.pick(file_num = 0, obj1 = 5))
        # print(make_data.place(obj1 = 5, obj2=6))
            
        # # Object 4
        # print(make_data.pick(file_num = 2, obj1 = 4))
        # print(make_data.place(obj1 = 4, obj2=6)) 

        # # Object 3
        # print(make_data.pick(file_num = 4, obj1 = 3))
        # print(make_data.place(obj1 = 3, obj2=6)) 

        # # Object 2
        # print(make_data.pick(file_num = 6, obj1 = 2))
        # print(make_data.place(obj1 = 2, obj2=6)) 

        # # Object 1
        # print(make_data.pick(file_num = 8, obj1 = 1))
        # print(make_data.place(obj1 = 1, obj2=6)) 

        # # Pour object 6 to 7
        # print(make_data.pour(file_num= 10, obj1= 6, obj2=7))



        # ### Checking graphs
        # # make_data.make_graph(fig_num=a, pos = position[a])
        # make_data.make_edge_index(i=a) 
        # # make_data.make_edge_attr(i=a) # def make_edge_index실행 후에 돌려

# plt.figure(figsize=)

### Checking paths

    # action_mix = ['pick','place','pick','place','pick','place','pick','place','pick','place','pour']
    action_v2_stacking = ['pick','place','pick','place','pick','place','pick','place']
    action_v3_stacking = ['pick','place','pick','place','pick','place']
    action_v4_stacking = ['pick','place','pick','place']
    ### action_v_mixing 고민을 더 해봐야 함! -> 덩이로 묶여서 실행이 되야함 

# print(make_data.pick(i=2, obj1=1))
# print(make_data.place(i=1, obj1=1, obj2=2))  # e.g.) obj1=3, obj2=4 -> obj1->obj2


# print(make_data.pick(i=0, obj1= 2))


# print(make_data.place(i=3,obj1=2, obj2=3))


# make_data.save_file(action='pick')
# make_data.save_file(action='place')





            
