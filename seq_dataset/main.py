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

    # # Velcro 2
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_2_3_45')
    # make_data = MakeDataset(problem = 'stacking_v2', example= 'ex_v2_1_23_4_5')

    # # Velcro 3
    # make_data = MakeDataset(problem = 'stacking_v3', example= 'ex_v3_123_4_5')
    # make_data = MakeDataset(problem = 'stacking_v3', example= 'ex_v3_1_234_5') 
    # make_data = MakeDataset(problem = 'stacking_v3', example= 'ex_v3_1_2_345') 

    # # Velcro 4
    # make_data = MakeDataset(problem = 'stacking_v4', example= 'v4_ex_1234_5')
    # make_data = MakeDataset(problem = 'stacking_v4', example= 'v4_ex_1_2345')

    # Mixing 2
    # make_data = MakeDataset(problem = 'mixing_v2', example= 'v2_ex_')


    # Original
    # make_data = MakeDataset(problem = 'mixing_5', example = 'mix_ex_1_2_3_4_5')
    # make_data = MakeDataset(problem = 'stacking_5', example = 'ex_1_2_3_4_5')

    # print(make_data.init_edge_attr(file_num= 13))
    ### Make initial edge index
    # make_data.init_edge_index()



    # make_data.changed_node_feature()
    
    # for a in range(0,1): # stacking (0,9) 0~8, mixing (0,14) 0~13
    # # #     ################### Call sample data ####################
    # # #         # action_seq = ['pick','place','pick','place','pick','place','pick','place','pick','place']
    #     print(f"\n[[[[[Task{a}]]]]]")
    #     # print(make_data.sample_data(i=a))
    #     #[Warning]#### 0으로 시작 # print(make_data.init_edge_attr(file_num = a))


    #     ### Checking graphs
        # make_data.make_graph(fig_num=a, pos = mix_pos[a])
        # make_data.make_graph(fig_num=a, pos = stack_pos[a])
        # make_data.make_edge_index(i=a) 
        # make_data.make_edge_attr(i=a) # def make_edge_index실행 후에 돌려
    #     action_v2_stacking = ['pick','place','pick','place','pick','place','pick','place']
    # ############################## [Stacking_v2] ##############################
    ## 1_2_34_5 # Attach - file 3
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_2_34_5')
    # print(make_data.init_node_features())
    # print(make_data.changed_node_feature())
    # print(make_data.init_edge_index())
    # print(make_data.pick(load_file_num= 0, obj1= 3))
    # print(make_data.place(load_file_num= 1, obj1=3, obj2=4))
    # print(make_data.attach(load_file_num = 2,obj1 = 3, obj2 = 4))
    # print(make_data.pick(load_file_num= 3, obj1= 34))
    # print(make_data.place(load_file_num= 4, obj1=34, obj2=5))
    # print(make_data.pick(load_file_num= 5, obj1= 2))
    # print(make_data.place(load_file_num= 6, obj1= 2, obj2= 34))
    # print(make_data.pick(load_file_num= 7, obj1= 1))
    # print(make_data.place(load_file_num= 8, obj1= 1, obj2= 2))
    
    # # ## 1_2_3_45 # Attach - file 3
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_2_3_45')
    # print(make_data.init_node_features())
    # print(make_data.changed_node_feature())
    # print(make_data.init_edge_index())
    # print(make_data.pick(load_file_num= 0, obj1= 4))
    # print(make_data.place(load_file_num= 1, obj1=4, obj2=5))
    # print(make_data.attach(load_file_num = 2,obj1 = 4, obj2 = 5))
    # print(make_data.pick(load_file_num= 3, obj1= 3))
    # print(make_data.place(load_file_num= 4, obj1=3, obj2=45))
    # print(make_data.pick(load_file_num= 5, obj1= 2))
    # print(make_data.place(load_file_num= 6, obj1= 2, obj2= 3))
    # print(make_data.pick(load_file_num= 7, obj1= 1))
    # print(make_data.place(load_file_num= 8, obj1= 1, obj2= 2))


    # ## 1_23_4_5 # Attach - file 5
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_23_4_5')
    # print(make_data.init_node_features())
    # print(make_data.changed_node_feature())
    # print(make_data.init_edge_index())
    # print(make_data.pick(load_file_num= 0, obj1= 4))
    # print(make_data.place(load_file_num= 1, obj1= 4, obj2= 5))
    # print(make_data.pick(load_file_num= 2, obj1= 2))
    # print(make_data.place(load_file_num= 3, obj1= 2, obj2= 3))
    # print(make_data.attach(load_file_num = 4,obj1 = 2, obj2 = 3))
    # print(make_data.pick(load_file_num= 5, obj1= 23))
    # print(make_data.place(load_file_num= 6, obj1= 23, obj2= 4))
    # print(make_data.pick(load_file_num= 7, obj1= 1))
    # print(make_data.place(load_file_num= 8, obj1= 1, obj2= 23))

    ## 12_3_4_5 # Attach - file 7
    # make_data = MakeDataset(problem = 'stacking_v2', example= '12_3_4_5')
    # print(make_data.init_node_features())
    # print(make_data.changed_node_feature())
    # print(make_data.init_edge_index())
    # print(make_data.pick(load_file_num= 0, obj1= 4))
    # print(make_data.place(load_file_num= 1, obj1= 4, obj2= 5))
    # print(make_data.pick(load_file_num= 2, obj1= 3))
    # print(make_data.place(load_file_num= 3, obj1= 3, obj2= 4))
    # print(make_data.pick(load_file_num= 4, obj1= 1))
    # print(make_data.place(load_file_num= 5, obj1= 1, obj2= 2))
    # print(make_data.attach(load_file_num = 6,obj1 = 1, obj2 = 2))
    # print(make_data.pick(load_file_num= 7, obj1= 12))
    # print(make_data.place(load_file_num= 8, obj1= 12, obj2= 3))


    ## 1_2_345 # Attach - file 3번과 6번에 2번 걸쳐 일어남
    # make_data = MakeDataset(problem='stacking_v3', example= '1_2_345')

    # # Node features
    # print(make_data.init_node_features())
    # print(make_data.same_node_features(save_num = 1))
    # print(make_data.same_node_features(save_num = 2))
    # print(make_data.changed_node_feature(save_num = 3))
    # print(make_data.same_node_features(save_num = 4))
    # print(make_data.same_node_features(save_num = 5))
    # print(make_data.changed_node_feature(save_num = 6))
    # print(make_data.same_node_features(save_num = 7))
    # print(make_data.same_node_features(save_num = 8))
    # print(make_data.same_node_features(save_num = 9))
    # print(make_data.same_node_features(save_num = 10))

    # # Edge index # 나중에는 번호마저 자동화하는 걸로!
    # print(make_data.init_edge_index())
    # print(make_data.pick_inx(save_num=1, obj1= 4))
    # print(make_data.place_inx(save_num=2, obj1=4, obj2= 5))
    # print(make_data.attach_inx(save_num=3, obj1=4, obj2= 5))
    # print(make_data.pick_inx(save_num=4, obj1=3))
    # print(make_data.place_inx(save_num=5, obj1=3, obj2= 45))
    # print(make_data.attach_inx(save_num=6, obj1=3, obj2= 45))
    # print(make_data.pick_inx(save_num=7, obj1=2))
    # print(make_data.place_inx(save_num=8, obj1=2, obj2=345))
    # print(make_data.pick_inx(save_num=9, obj1=1))
    # print(make_data.place_inx(save_num=10, obj1=1, obj2=2))

    # # # Edge attribute
    # print(make_data.init_edge_attr()) # 뒤에 노드 8번이 들어가면 'rel_on_right' = 1, 앞에 node 8번이 들어가면 'rel_on_left'로 지정
    # print(make_data.pick_attr(save_num=1)) # 
    # print(make_data.place_attr(save_num=2, obj1=4, obj2=5))
    # print(make_data.attach_attr(save_num=3, obj1=4, obj2=5))
    # print(make_data.pick_attr(save_num=4)) 
    # print(make_data.place_attr(save_num=5, obj1=3, obj2=45))
    # print(make_data.attach_attr(save_num=6, obj1=3, obj2= 45))
    # print(make_data.pick_attr(save_num=7))
    # print(make_data.place_attr(save_num=8, obj1=2, obj2=345))
    # print(make_data.pick_attr(save_num=9))
    # print(make_data.place_attr(save_num=10, obj1=1, obj2=2))

    # make_data = MakeDataset(problem='stacking_v3', example= '1_234_5')

    # # Node features
    # print(make_data.init_node_features())
    # print(make_data.same_node_features(save_num = 1))
    # print(make_data.same_node_features(save_num = 2))
    # print(make_data.changed_node_feature(save_num = 3))
    # print(make_data.same_node_features(save_num = 4))
    # print(make_data.same_node_features(save_num = 5))
    # print(make_data.changed_node_feature(save_num = 6))
    # print(make_data.same_node_features(save_num = 7))
    # print(make_data.same_node_features(save_num = 8))
    # print(make_data.same_node_features(save_num = 9))
    # print(make_data.same_node_features(save_num = 10))

    # # Edge index # 나중에는 번호마저 자동화하는 걸로!
    # print(make_data.init_edge_index())
    # print(make_data.pick_inx(save_num=1, obj1= 3))
    # print(make_data.place_inx(save_num=2, obj1=3, obj2= 4))
    # print(make_data.attach_inx(save_num=3, obj1=3, obj2= 4))
    # print(make_data.pick_inx(save_num=4, obj1=2))
    # print(make_data.place_inx(save_num=5, obj1=2, obj2= 34))
    # print(make_data.attach_inx(save_num=6, obj1=2, obj2= 34))
    # print(make_data.pick_inx(save_num=7, obj1=234))
    # print(make_data.place_inx(save_num=8, obj1=234, obj2=5))
    # print(make_data.pick_inx(save_num=9, obj1=1))
    # print(make_data.place_inx(save_num=10, obj1=1, obj2=234))

    # # # Edge attribute
    # print(make_data.init_edge_attr()) # 뒤에 노드 8번이 들어가면 'rel_on_right' = 1, 앞에 node 8번이 들어가면 'rel_on_left'로 지정
    # print(make_data.pick_attr(save_num=1))
    # print(make_data.place_attr(save_num=2, obj1=3, obj2= 4))
    # print(make_data.attach_attr(save_num=3, obj1=3, obj2= 4))
    # print(make_data.pick_attr(save_num=4)) # 노드 0번에 들린 node로 계산
    # print(make_data.place_attr(save_num=5, obj1=2, obj2= 34))
    # print(make_data.attach_attr(save_num=6, obj1=2, obj2= 34))
    # print(make_data.pick_attr(save_num=7))
    # print(make_data.place_attr(save_num=8, obj1=234, obj2=5))
    # print(make_data.pick_attr(save_num=9))
    # print(make_data.place_attr(save_num=10, obj1=1, obj2=234))

############################### [Stacking 123_4_5] ################################################
    make_data = MakeDataset(problem='stacking_v3', example= '123_4_5')
    
    # Node features
    # print(make_data.init_node_features())
    # print(make_data.same_node_features(save_num = 1))
    # print(make_data.same_node_features(save_num = 2))
    # print(make_data.same_node_features(save_num = 3))
    # print(make_data.same_node_features(save_num = 4))
    # print(make_data.changed_node_feature(save_num = 5))
    # print(make_data.same_node_features(save_num = 6))
    # print(make_data.same_node_features(save_num = 7))
    # print(make_data.changed_node_feature(save_num = 8))
    # print(make_data.same_node_features(save_num = 9))
    # print(make_data.same_node_features(save_num = 10))

    # Edge index # 나중에는 번호마저 자동화하는 걸로!
    # print(make_data.init_edge_index())
    # print(make_data.pick_inx(save_num=1, obj1= 4))
    # print(make_data.place_inx(save_num=2, obj1=4, obj2= 5))
    # print(make_data.pick_inx(save_num=3, obj1=2))
    # print(make_data.place_inx(save_num=4, obj1=2, obj2= 3))
    # print(make_data.attach_inx(save_num=5, obj1=2, obj2= 3))
    # print(make_data.pick_inx(save_num=6, obj1=1))
    # print(make_data.place_inx(save_num=7, obj1=1, obj2=23))
    # print(make_data.attach_inx(save_num=8, obj1=1, obj2= 23))
    # print(make_data.pick_inx(save_num=9, obj1=123))
    # print(make_data.place_inx(save_num=10, obj1=123, obj2=4))

    # # Edge attribute
    print(make_data.init_edge_attr()) # 뒤에 노드 8번이 들어가면 'rel_on_right' = 1, 앞에 node 8번이 들어가면 'rel_on_left'로 지정
    print(make_data.pick_attr(save_num=1))
    # print(make_data.place_attr(save_num=2, obj1=4, obj2= 5))
    # print(make_data.pick_attr(save_num=3))
    # print(make_data.place_attr(save_num=4, obj1=2, obj2= 3))
    # print(make_data.attach_attr(save_num=5, obj1=2, obj2= 3))
    # print(make_data.pick_attr(save_num=6))
    # print(make_data.place_attr(save_num=7, obj1=1, obj2=23))
    # print(make_data.attach_attr(save_num=8, obj1=1, obj2= 23))
    # print(make_data.pick_attr(save_num=9))
    # print(make_data.place_attr(save_num=10, obj1=123, obj2=4))



############################### [Stacking 1234_5] ################################################
    # make_data = MakeDataset(problem='stacking_v4', example= '1234_5')
    # # Node features
    # print(make_data.init_node_features())
    # print(make_data.same_node_features(save_num = 1))
    # print(make_data.same_node_features(save_num = 2))
    # print(make_data.changed_node_feature(save_num = 3))
    # print(make_data.same_node_features(save_num = 4))
    # print(make_data.same_node_features(save_num = 5))
    # print(make_data.changed_node_feature(save_num = 6))
    # print(make_data.same_node_features(save_num = 7))
    # print(make_data.same_node_features(save_num = 8))
    # print(make_data.changed_node_feature(save_num = 9))
    # print(make_data.same_node_features(save_num = 10))
    # print(make_data.same_node_features(save_num = 11))

    # # Edge index # 나중에는 번호마저 자동화하는 걸로!
    # print(make_data.init_edge_index())
    # print(make_data.pick_inx(save_num=1, obj1= 3))
    # print(make_data.place_inx(save_num=2, obj1=3, obj2= 4))
    # print(make_data.attach_inx(save_num=3, obj1= 3, obj2= 4))
    # print(make_data.pick_inx(save_num=4, obj1=2))
    # print(make_data.place_inx(save_num=5, obj1=2, obj2= 34))
    # print(make_data.attach_inx(save_num=6, obj1=2, obj2= 34))
    # print(make_data.pick_inx(save_num=7, obj1=1))
    # print(make_data.place_inx(save_num=8, obj1=1, obj2=234))
    # print(make_data.attach_inx(save_num=9, obj1=1, obj2= 234))
    # print(make_data.pick_inx(save_num=10, obj1=1234))
    # print(make_data.place_inx(save_num=11, obj1=1234, obj2=5))

    # # # Edge attribute
    # print(make_data.init_edge_attr()) # 뒤에 노드 8번이 들어가면 'rel_on_right' = 1, 앞에 node 8번이 들어가면 'rel_on_left'로 지정
    # print(make_data.pick_attr(save_num=1))
    # print(make_data.place_attr(save_num=2, obj1=3, obj2= 4))
    # print(make_data.attach_attr(save_num=3, obj1= 3, obj2= 4))
    # print(make_data.pick_attr(save_num=4))
    # print(make_data.place_attr(save_num=5, obj1=2, obj2= 34))
    # print(make_data.attach_attr(save_num=6, obj1=2, obj2= 34))
    # print(make_data.pick_attr(save_num=7))
    # print(make_data.place_attr(save_num=8, obj1=1, obj2=234))
    # print(make_data.attach_attr(save_num=9, obj1=1, obj2= 234))
    # print(make_data.pick_attr(save_num=10))
    # print(make_data.place_attr(save_num=11, obj1=1234, obj2=5))

############################### [Stacking 1_2345] ################################################
    # make_data = MakeDataset(problem='stacking_v4', example= '1_2345')
    # # Node features
    # print(make_data.init_node_features())
    # print(make_data.same_node_features(save_num = 1))
    # print(make_data.same_node_features(save_num = 2))
    # print(make_data.changed_node_feature(save_num = 3))
    # print(make_data.same_node_features(save_num = 4))
    # print(make_data.same_node_features(save_num = 5))
    # print(make_data.changed_node_feature(save_num = 6))
    # print(make_data.same_node_features(save_num = 7))
    # print(make_data.same_node_features(save_num = 8))
    # print(make_data.changed_node_feature(save_num = 9))
    # print(make_data.same_node_features(save_num = 10))
    # print(make_data.same_node_features(save_num = 11))

    # # Edge index # 나중에는 번호마저 자동화하는 걸로!
    # print(make_data.init_edge_index())
    # print(make_data.pick_inx(save_num=1, obj1= 4))
    # print(make_data.place_inx(save_num=2, obj1=4, obj2= 5))
    # print(make_data.attach_inx(save_num=3, obj1= 4, obj2= 5))
    # print(make_data.pick_inx(save_num=4, obj1=3))
    # print(make_data.place_inx(save_num=5, obj1=3, obj2= 45))
    # print(make_data.attach_inx(save_num=6, obj1=3, obj2= 45))
    # print(make_data.pick_inx(save_num=7, obj1=2))
    # print(make_data.place_inx(save_num=8, obj1=2, obj2=345))
    # print(make_data.attach_inx(save_num=9, obj1=2, obj2= 345))
    # print(make_data.pick_inx(save_num=10, obj1=1))
    # print(make_data.place_inx(save_num=11, obj1=1, obj2=2345))

    # # # Edge attribute
    # print(make_data.init_edge_attr()) # 뒤에 노드 8번이 들어가면 'rel_on_right' = 1, 앞에 node 8번이 들어가면 'rel_on_left'로 지정
    # print(make_data.pick_attr(save_num=1))
    # print(make_data.place_attr(save_num=2, obj1=4, obj2= 5))
    # print(make_data.attach_attr(save_num=3, obj1= 4, obj2= 5))
    # print(make_data.pick_attr(save_num=4))
    # print(make_data.place_attr(save_num=5, obj1=3, obj2= 45))
    # print(make_data.attach_attr(save_num=6, obj1=3, obj2= 45))
    # print(make_data.pick_attr(save_num=7))
    # print(make_data.place_attr(save_num=8, obj1=2, obj2=345))
    # print(make_data.attach_attr(save_num=9, obj1=2, obj2= 345))
    # print(make_data.pick_attr(save_num=10))
    # print(make_data.place_attr(save_num=11, obj1=1, obj2=2345))

    ############################## [Mixing_5] ##############################
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

    # # Pick bowl1   
    # print(make_data.pick(file_num=10, obj1=6))

    #  # # Pour object 6 to 7
    # print(make_data.pour(file_num= 11, obj1= 6, obj2=7))

    # ## Place bowl1 on the table
    # print(make_data.place(obj1=6, obj2=8, load_file_num=12, save_file_num=13))

    ######################################################################

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





            
