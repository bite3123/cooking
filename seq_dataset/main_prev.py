from seq_dataset.seq_demo_prev import *


# def node_features(end, *attach_num):
def node_features(end):
    for i in range(0, end): 
        if i == 0:
            print(make_data.init_node_features())
        else:
            print(make_data.same_node_features(n=i))
            
                
def edge_index(list_inx):
    for i, (n, obj1, obj2) in enumerate(list_inx):
        if i == 0:
            print(make_data.init_edge_index())
        if obj2 == None:
            print(make_data.pick_inx(n=n, obj1=obj1))
        else:
            print(make_data.place_inx(n=n, obj1=obj1, obj2=obj2))


def edge_attr(list_attr,num):
    for i, (n, obj1, obj2) in enumerate(list_attr):
        if i == 0:
            print(make_data.init_edge_attr(n, num))
        if obj2 == None:
            print(make_data.pick_attr(n=n, obj1=obj1)) 
        else:
            print(make_data.place_attr(n=n, obj1=obj1, obj2=obj2))




if __name__ == '__main__':
    # ################################################### [Stacking_5 / 1_2_3_4_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_5', example= '1_2_3_4_5')

    
    # stack_normal = [(1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box3', None), (4, 'Box3', 'Box4'), (5, 'Box2', None), \
    #                 (6, 'Box2', 'Box3'), (7, 'Box1', None), (8, 'Box1', 'Box2')]
  
    # node_features(9)
    # edge_index(stack_normal)
    # edge_attr(stack_normal,9)

    
    # ################################################## [Stacking_v2 / 1_2_3_45] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_2_3_45')
    # attach_45 = [(1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box3', None), (4, 'Box3', 'Box4 + Box5'), \
    #             (5, 'Box2', None), (6, 'Box2', 'Box3'), (7, 'Box1', None), (8, 'Box1', 'Box2')]

    # node_features(9)
    # edge_index(attach_45)
    # edge_attr(attach_45, 9)
    #  ################################################## [Stacking_v2 / 1_2_34_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_2_34_5')
    # attach_34 = [(1, 'Box3', None), (2, 'Box3', 'Box4'), (3, 'Box3 + Box4', None), (4, 'Box3 + Box4', 'Box5'), \
    #             (5, 'Box2', None), (6, 'Box2', 'Box3 + Box4'), (7, 'Box1', None), (8, 'Box1', 'Box2')]
    
    # node_features(9)
    # edge_index(attach_34)
    # edge_attr(attach_34, 9)
    #  ################################################## [Stacking_v2 / 1_23_4_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_23_4_5')
    # attach_23 = [(1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box2', None), (4, 'Box2', 'Box3'), (5, 'Box2 + Box3', None),\
    #             (6, 'Box2 + Box3', 4), (7, 'Box1', None), (8, 'Box1', 'Box2 + Box3')]
    
    # node_features(9)
    # edge_index(attach_23)
    # edge_attr(attach_23, 9)
    #  ################################################## [Stacking_v2 / 12_3_4_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '12_3_4_5')
    # attach_12 = [(1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box3', None), (4, 'Box3', 'Box4'),\
    #             (5, 'Box1', None), (6, 'Box1', 'Box2'), (7, 'Box1 + Box2', None), (8, 'Box1 + Box2', 'Box3')]
    
    # node_features(9)
    # edge_index(attach_12)
    # edge_attr(attach_12, 9)
     ################################################### [Stacking_v2 / 12_3_45] ########################################################
     # 2개씩 붙는 것도 있겠다
     
     ################################################### [Stacking_v3 / 1_2_345] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v3', example= '1_2_345')
    # attach_345 = [(1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box3', None), (4, 'Box3', 'Box4 + Box5'),\
    #             (5, 'Box2', None), (6, 'Box2', 'Box3 + Box4 + Box5'), (7, 'Box1', None), (8, 'Box1', 'Box2')]
    
    
    # # node_features(9)
    # edge_index(attach_345)
    # edge_attr(attach_345,9)

    # attach_345 
     ################################################### [Stacking_v3 / 1_234_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v3', example= '1_234_5')
    # attach_345 = [(1, 'Box3', None), (2, 'Box3', 'Box4'), (3, 'Box2', None), (4, 'Box2', 'Box3 + Box4'),\
    #             (5, 'Box2 + Box3 + Box4', None), (6, 'Box2 + Box3 + Box4', 'Box5'), (7, 'Box1', None), (8, 'Box1', 'Box2 + Box3 + Box4')]
    
    
    # node_features(9)
    # edge_index(attach_345)
    # edge_attr(attach_345, 9)
     ################################################### [Stacking_v3 / 123_4_5] ########################################################
    make_data = MakeDataset(problem = 'stacking_v3', example= '123_4_5')
    attach_345 = [(1, 'Box4', None), (2, 'Box4', 5), (3, 'Box2', None), (4, 'Box2', 'Box3'),  (5, 'Box1', None), (6, 'Box1', 'Box2 + Box3'),\
                (7, 'Box1 + Box2 + Box3', None), (8, 'Box1 + Box2 + Box3', 'Box4')]
    
    # node_features(9)
    # edge_index(attach_345)
    # edge_attr(attach_345, 9)
     ################################################### [Stacking_v4 / 1_2345] ########################################################
    make_data = MakeDataset(problem = 'stacking_v4', example= '1_2345')
    attach_345 = [(1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box3', None), (4, 'Box3', 'Box4 + Box5'), (5, 'Box2', None), \
                  (6, 'Box2', 'Box3 + Box4 + Box5'), (7, 'Box1', None), (8, 'Box1', 'Box2 + Box3 + Box4 + Box5')]
    
    node_features(9)
    edge_index(attach_345)
    edge_attr(attach_345,9)
     ################################################### [Stacking_v4 / 1234_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v4', example= '1234_5')
    # attach_345 = [(1, 3, None), (2, 3, 4), (3, 3, 4), (4, 2, None), (5, 2, 34), (6, 2 ,34), (7, 1, None), \
    #               (8, 1, 234), (9, 1, 234), (10, 1234, None), (11, 1234, 5)]
    
    # node_features(1)
    # edge_index(attach_345, 3,6,9)
    # edge_attr(attach_345, 3,6,9)
    
    ################################################### [Mixing_v2 / 1_2_3_45] ########################################################
    # make_data = MakeDataset(problem = 'mixing_v2', example= '1_2_3_45') 
    
    # attach_45 = [(1, 4, None), (2, 4, 5), (3, 4, 5), (4, 3, None), (5, 3, 45), (6, 2, None), (7, 2, 3), (8, 1, None), (9, 1, 2)]
  
    # node_features(10, 3)
    # edge_index(attach_45, 3)
    # edge_attr(attach_45, 3)



            