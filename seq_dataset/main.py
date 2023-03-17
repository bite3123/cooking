from seq_demo import *


# def node_features(end, *attach_num):
def node_features(end):
    for i in range(0, end): 
        print(make_data.same_node_features(n=i))
        # if i == 0:
        #     print(make_data.init_node_features())
        # elif i in attach_num:
        #     print(make_data.changed_node_feature(n=i))
        # else:
        #     print(make_data.same_node_features(n=i))
            
                
def edge_index(list_inx, *attach_num):
    for i, (n, obj1, obj2) in enumerate(list_inx):
        if i == 0:
            print(make_data.init_edge_index())
        # if n not in attach_num and obj2 == None:
        #     print(make_data.pick_inx(n=n, obj1=obj1))
        # elif n in attach_num:
        #     print(make_data.attach_inx(n=n, obj1=obj1, obj2=obj2))
        if obj2 == None:
            print(make_data.pick_inx(n=n, obj1=obj1))
        else:
            print(make_data.place_inx(n=n, obj1=obj1, obj2=obj2))


def edge_attr(list_attr, *attach_num):
    for i, (n, obj1, obj2) in enumerate(list_attr):
        if i == 0:
            print(make_data.init_edge_attr(n))
        if n not in attach_num and obj2 == None:
            print(make_data.pick_attr(n=n, obj1=obj1)) 
        elif n in attach_num:
            print(make_data.attach_attr(n=n, obj1=obj1, obj2=obj2))
        else:
            print(make_data.place_attr(n=n, obj1=obj1, obj2=obj2))


# def node_features_normal(end):
#     for i in range(0, end): 
#         if i == 0:
#             print(make_data.init_node_features())
#         else:
#             print(make_data.same_node_features(n=i))
            
                
# def edge_index_normal(list_inx):
#     for i, (n, obj1, obj2) in enumerate(list_inx):
#         if i == 0:
#             print(make_data.init_edge_index())
#         if obj2 == None:
#             print(make_data.pick_inx(n=n, obj1=obj1))
#         else:
#             print(make_data.place_inx(n=n, obj1=obj1, obj2=obj2))

# def edge_attr_normal(list_attr):
#     for i, (n, obj1, obj2) in enumerate(list_attr):
#         if i == 0:
#             print(make_data.init_edge_attr())
#         if obj2 == None:
#             print(make_data.pick_attr(n=n, obj1=obj1)) 
#         else:
#             print(make_data.place_attr(n=n, obj1=obj1, obj2=obj2))


if __name__ == '__main__':
    # ################################################### [Stacking_5 / 1_2_3_4_5] ########################################################
    make_data = MakeDataset(problem = 'stacking_5', example= '1_2_3_4_5')

    
    # stack_normal = [(1, 4, None), (2, 4, 5), (3, 3, None), (4, 3, 4), (5, 2, None), (6, 2, 3), (7, 1, None), (8, 1, 2)]
  
    # node_features(9)
    # edge_index(stack_normal, None)
    # edge_attr(stack_normal, None)

    
    ################################################### [Stacking_v2 / 1_2_3_45] ########################################################
    make_data = MakeDataset(problem = 'stacking_v2', example= '1_2_3_45')
    attach_45 = [(1, 4, None), (2, 4, 5), (3, 4, 5), (4, 3, None), (5, 3, 45), (6, 2, None), (7, 2, 3), (8, 1, None), (9, 1, 2)]

    # node_features(10)
    edge_index(attach_45, 3)
    #
    # edge_attr(attach_45, 3)
     ################################################### [Stacking_v2 / 1_2_34_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_2_34_5')
    # attach_34 = [(1, 3, None), (2, 3, 4), (3, 3, 4), (4, 34, None), (5, 34, 5), (6, 2, None), (7, 2, 34), (8, 1, None), (9, 1, 2)]
    
    # node_features(10, 3)
    # edge_index(attach_34, 3)
    # edge_attr(attach_34, 3)
     ################################################### [Stacking_v2 / 1_23_4_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_23_4_5')
    # attach_23 = [(1, 4, None), (2, 4, 5), (3, 2, None), (4, 2, 3), (5, 2, 3), (6, 23, None), (7, 23, 4), (8, 1, None), (9, 1, 23)]
    
    # node_features(10, 5)
    # edge_index(attach_23, 5)
    # edge_attr(attach_23, 5)
     ################################################### [Stacking_v2 / 12_3_4_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '12_3_4_5')
    # attach_12 = [(1, 4, None), (2, 4, 5), (3, 3, None), (4, 3, 4), (5, 1, None), (6, 1, 2), (7, 1, 2), (8, 12, None), (9, 12, 3)]
    
    # node_features(10, 7)
    # edge_index(attach_12, 7)
    # edge_attr(attach_12, 7)
     ################################################### [Stacking_v2 / 12_3_45] ########################################################
     # 2개씩 붙는 것도 있겠다
     
     ################################################### [Stacking_v3 / 1_2_345] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v3', example= '1_2_345')
    # attach_345 = [(1, 4, None), (2, 4, 5), (3, 4, 5), (4, 3, None), (5, 3, 45), (6, 3, 45), (7, 2, None), (8, 2, 345), (9, 1, None), (10, 1, 2)]
    
    
    # node_features(11, 3,6)
    # edge_index(attach_345, 3,6)
    # edge_attr(attach_345, 3,6)

    # attach_345 
     ################################################### [Stacking_v3 / 1_234_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v3', example= '1_234_5')
    # attach_345 = [(1, 3, None), (2, 3, 4), (3, 3, 4), (4, 2, None), (5, 2, 34), (6, 2, 34), (7, 234, None), (8, 234, 5), (9, 1, None), (10, 1, 234)]
    
    
    # node_features(11, 3,6)
    # edge_index(attach_345, 3,6)
    # edge_attr(attach_345, 3,6)
     ################################################### [Stacking_v3 / 123_4_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v3', example= '123_4_5')
    # attach_345 = [(1, 4, None), (2, 4, 5), (3, 2, None), (4, 2, 3), (5, 2, 3), (6, 1, None), (7, 1, 23), (8, 1, 23), (9, 123, None), (10, 123, 4)]
    
    # node_features(11, 5,8)
    # edge_index(attach_345, 5,8)
    # edge_attr(attach_345, 5,8)
     ################################################### [Stacking_v4 / 1_2345] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v4', example= '1_2345')
    # attach_345 = [(1, 4, None), (2, 4, 5), (3, 4, 5), (4, 3, None), (5, 3, 45), (6, 3 ,45), (7, 2, None), \
    #               (8, 2, 345), (9, 2, 345), (10, 1, None), (11, 1, 2345)]
    
    # node_features(12, 3,6,9)
    # edge_index(attach_345, 3,6,9)
    # edge_attr(attach_345, 3,6,9)
     ################################################### [Stacking_v4 / 1234_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v4', example= '1234_5')
    # attach_345 = [(1, 3, None), (2, 3, 4), (3, 3, 4), (4, 2, None), (5, 2, 34), (6, 2 ,34), (7, 1, None), \
    #               (8, 1, 234), (9, 1, 234), (10, 1234, None), (11, 1234, 5)]
    
    # node_features(12, 3,6,9)
    # edge_index(attach_345, 3,6,9)
    # edge_attr(attach_345, 3,6,9)
    
    ################################################### [Mixing_v2 / 1_2_3_45] ########################################################
    # make_data = MakeDataset(problem = 'mixing_v2', example= '1_2_3_45') 
    
    # attach_45 = [(1, 4, None), (2, 4, 5), (3, 4, 5), (4, 3, None), (5, 3, 45), (6, 2, None), (7, 2, 3), (8, 1, None), (9, 1, 2)]
  
    # node_features(10, 3)
    # edge_index(attach_45, 3)
    # edge_attr(attach_45, 3)



            