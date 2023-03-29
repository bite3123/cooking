from seq_demo import *


# class stacking_problem():
def stack_node_features(list):
    end = len(list)
    for i in range(0, end): 
        if i == 0:
            print(make_data.init_node_features())
        else:
            print(make_data.same_node_features(n=i))

            
                
def stack_edge_index(list_inx):
    for i, (n, obj1, obj2) in enumerate(list_inx):
        if i == 0:
            print(make_data.init_edge_index())
        elif obj2 == None:
            print(make_data.pick_inx(n=n, obj1=obj1))
        else:
            print(make_data.place_inx(n=n, obj1=obj1, obj2=obj2))


def stack_edge_attr(list):
    int = len(list)
    for i, (n, obj1, obj2) in enumerate(list):
        if i == 0:
            print(make_data.init_edge_attr(n, int))
        elif obj2 == None:
            print(make_data.pick_attr(n=n, obj1=obj1)) 
        else:
            print(make_data.place_attr(n=n, obj1=obj1, obj2=obj2))



# Graph의 node color 
def make_node_colors(list,i):
    if list[i][0] == 0:
        node1 = 'Table'
        node2 = None
        
    else:
        node1 = list[i][1]
        if list[i][2] != None:
            node2 = list[i][2]
        else:
            node2 = 'Robot_hand'
    return node1, node2

def stack_main(list):
    num = len(list)
    stack_node_features(list)
    stack_edge_index(list)
    stack_edge_attr(list)

    for i in range(num):
        print(f"====================================================[Task{i}]====================================================")
        node1, node2 = make_node_colors(list,i)
        make_data.check_graph(i, node1, node2)
    print("[[[Graph end]]]")

########################################################### [Mix] ########################################################################
def mix_node_features(list):
    end = len(list)
    for i in range(0, end): 
        if i == 0:
            print(make_data.init_node_features())
        else:
            print(make_data.same_node_features(n=i))

            
                
def mix_edge_index(list):
    for i, (n, obj1, obj2) in enumerate(list):
        if i == 0:
            print(make_data.init_edge_index())
        elif obj2 == None:
            print(make_data.pick_inx(n=n, obj1=obj1))
        elif obj2 == 'Table':
            print(make_data.place_inx(n=n, obj1=obj1, obj2= 'Table'))
        else:
            if "Bowl" in obj1 and "Bowl" in obj2:
                print(make_data.pour_inx(n=n, obj1=obj1, obj2 = obj2))
            elif "Box" in obj1 and "Box" in obj2:
                print(make_data.mix_inx(n=n, obj1=obj1, obj2 = obj2))
            else:
                print("Wrong task")


def mix_edge_attr(list):
    for i, (n, obj1, obj2) in enumerate(list):
        int = len(list)
        if i == 0:
            print(make_data.init_edge_attr(n, int))
        elif obj2 == None:
            print(make_data.pick_attr(n=n, obj1=obj1)) 
        elif obj2 == 'Table':
            print(make_data.place_attr(n=n, obj1=obj1, obj2=obj2))
        else:
            if "Bowl" in obj1 and "Bowl" in obj2:
                print(make_data.pour_attr(n=n, obj1=obj1, obj2 = obj2))
            elif "Box" in obj1 and "Box" in obj2:
                print(make_data.mix_attr(n=n, obj1=obj1, obj2 = obj2))
            else:
                print("Wrong task")
            
def mix_main(list):
    num = len(list)
    mix_node_features(list)
    mix_edge_index(list)
    mix_edge_attr(list)

    for i in range(num):
        print(f"====================================================[Task{i}]====================================================")
        node1, node2 = make_node_colors(list,i)
        make_data.check_graph(i, node1, node2)
    print("[[[Graph end]]]")
        

if __name__ == '__main__':
    # ################################################### [Stacking_5 / 1_2_3_4_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_5', example= '1_2_3_4_5')

    # stack = [(0, None, None), (1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box3', None), (4, 'Box3', 'Box4'), (5, 'Box2', None), \
    #                 (6, 'Box2', 'Box3'), (7, 'Box1', None), (8, 'Box1', 'Box2')]
    # stack_main(stack)
    
    ################################################## [Stacking_v2 / 1_2_3_45] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_2_3_45')
    # stack = [(0, None, None), (1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box3', None), (4, 'Box3', 'Box4 + Box5'), \
    #             (5, 'Box2', None), (6, 'Box2', 'Box3'), (7, 'Box1', None), (8, 'Box1', 'Box2')]
    # stack_main(stack)

    #  ################################################## [Stacking_v2 / 1_2_34_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_2_34_5')
    # stack = [(0, None, None), (1, 'Box3', None), (2, 'Box3', 'Box4'), (3, 'Box3 + Box4', None), (4, 'Box3 + Box4', 'Box5'), \
    #             (5, 'Box2', None), (6, 'Box2', 'Box3 + Box4'), (7, 'Box1', None), (8, 'Box1', 'Box2')]
    
    # stack_main(stack)
  
     ################################################## [Stacking_v2 / 1_23_4_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '1_23_4_5')
    # stack = [(0, None, None), (1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box2', None), (4, 'Box2', 'Box3'), (5, 'Box2 + Box3', None),\
    #             (6, 'Box2 + Box3', 'Box4'), (7, 'Box1', None), (8, 'Box1', 'Box2 + Box3')]
    # stack_main(stack)

   
    #  ################################################## [Stacking_v2 / 12_3_4_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v2', example= '12_3_4_5')
    # stack = [(0, None, None), (1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box3', None), (4, 'Box3', 'Box4'),\
    #             (5, 'Box1', None), (6, 'Box1', 'Box2'), (7, 'Box1 + Box2', None), (8, 'Box1 + Box2', 'Box3')]
    # stack_main(stack)


    #  ################################################## [Stacking_v2 / 12_3_45] ########################################################
    # ##  2개씩 붙는 것도 있겠다
     
    #  ################################################## [Stacking_v3 / 1_2_345] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v3', example= '1_2_345')
    # stack = [(0, None, None), (1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box3', None), (4, 'Box3', 'Box4 + Box5'),\
    #             (5, 'Box2', None), (6, 'Box2', 'Box3 + Box4 + Box5'), (7, 'Box1', None), (8, 'Box1', 'Box2')]
    # stack_main(stack)

    #  ################################################## [Stacking_v3 / 1_234_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v3', example= '1_234_5')
    # stack = [(0, None, None), (1, 'Box3', None), (2, 'Box3', 'Box4'), (3, 'Box2', None), (4, 'Box2', 'Box3 + Box4'),\
    #             (5, 'Box2 + Box3 + Box4', None), (6, 'Box2 + Box3 + Box4', 'Box5'), (7, 'Box1', None), (8, 'Box1', 'Box2 + Box3 + Box4')]
    # stack_main(stack)
  
    # #  ################################################## [Stacking_v3 / 123_4_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v3', example= '123_4_5')
    # stack = [(0, None, None), (1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box2', None), (4, 'Box2', 'Box3'),  (5, 'Box1', None),\
    #         (6, 'Box1', 'Box2 + Box3'), (7, 'Box1 + Box2 + Box3', None), (8, 'Box1 + Box2 + Box3', 'Box4')]
    # stack_main(stack)
    
    #  ################################################## [Stacking_v4 / 1_2345] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v4', example= '1_2345')
    # stack = [(0, None, None), (1, 'Box4', None), (2, 'Box4', 'Box5'), (3, 'Box3', None), (4, 'Box3', 'Box4 + Box5'), (5, 'Box2', None), \
    #               (6, 'Box2', 'Box3 + Box4 + Box5'), (7, 'Box1', None), (8, 'Box1', 'Box2 + Box3 + Box4 + Box5')]
    # stack_main(stack)
  
    #  ################################################## [Stacking_v4 / 1234_5] ########################################################
    # make_data = MakeDataset(problem = 'stacking_v4', example= '1234_5')
    # stack = [(0, None, None), (1, 'Box3', None), (2, 'Box3', 'Box4'), (3, 'Box2', None), (4, 'Box2', 'Box3 + Box4'),\
    #               (5, 'Box1', None), (6, 'Box1', 'Box2 + Box3 + Box4'),  \
    #               (7, 'Box1 + Box2 + Box3 + Box4', None), (8, 'Box1 + Box2 + Box3 + Box4', 'Box5')]
    
    # stack_main(stack)
    

#################################################################[MIX]##########################################################################

    # ################################################### [Mixing_5 / 1_2_3_4_5] ########################################################
    # make_data = MakeDataset(problem = 'mixing_5', example= '1_2_3_4_5') 
    # ## Action sequence example) pick, pour, place, pick, pour, place, pick, pour, place,....
    
    # mix = [(0, None, None), (1, 'Bowl5', None), (2, 'Bowl5', 'Bowl6'), (3, 'Bowl5', 'Table'), (4, 'Bowl4', None), (5, 'Bowl4', 'Bowl6'), \
    #         (6, 'Bowl4', 'Table'), (7, 'Bowl3', None), (8, 'Bowl3', 'Bowl6'), (9, 'Bowl3', 'Table'), (10, 'Bowl2', None), (11, 'Bowl2', 'Bowl6'), \
    #         (12, 'Bowl2', 'Table'), (13, 'Bowl1', None), (14,'Bowl1','Bowl6'), (15, 'Bowl1', 'Table') , (16, 'Bowl6', 'Bowl7')]
    
    # mix_main(mix)


    # ################################################### [Mixing_v2 / 1_2_3_45] ########################################################
    # make_data = MakeDataset(problem = 'mixing_v2', example= '1_2_3_45') 
    # # Action sequence example) pick, pour, place, pick, pour, place, mix, pick, pour, place,....
    # mix = [(0, None, None), (1, 'Bowl5', None), (2, 'Bowl5', 'Bowl6'), (3, 'Bowl5', 'Table'), (4, 'Bowl4', None), (5, 'Bowl4', 'Bowl6'), \
    #         (6, 'Bowl4', 'Table'), (7, 'Box4','Box5'), (8, 'Bowl3', None), (9, 'Bowl3', 'Bowl6'), (10, 'Bowl3', 'Table'), (11, 'Bowl2', None),\
    #        (12, 'Bowl2', 'Bowl6'), (13, 'Bowl2', 'Table'), (14, 'Bowl1', None), (15,'Bowl1','Bowl6'), (16, 'Bowl1', 'Table') , (17, 'Bowl6', 'Bowl7')]

    # mix_main(mix)

    ################################################### [Mixing_v2 / 1_2_34_5] ########################################################
    # make_data = MakeDataset(problem = 'mixing_v2', example= '1_2_34_5') 
    
    # mix = [(0, None, None), (1, 'Bowl4', None), (2, 'Bowl4', 'Bowl6'), (3, 'Bowl4', 'Table'), (4, 'Bowl3', None), (5, 'Bowl3', 'Bowl6'), \
    #         (6, 'Bowl3', 'Table'), (7, 'Box3','Box4'), (8, 'Bowl5', None), (9, 'Bowl5', 'Bowl6'), (10, 'Bowl5', 'Table'), (11, 'Bowl2', None),\
    #        (12, 'Bowl2', 'Bowl6'), (13, 'Bowl2', 'Table'), (14, 'Bowl1', None), (15,'Bowl1','Bowl6'), (16, 'Bowl1', 'Table') , (17, 'Bowl6', 'Bowl7')]

    # mix_main(mix)

    # ################################################### [Mixing_v2 / 1_23_4_5] ########################################################
    # make_data = MakeDataset(problem = 'mixing_v2', example= '1_23_4_5') 
    
    # mix = [(0, None, None), (1, 'Bowl3', None), (2, 'Bowl3', 'Bowl6'), (3, 'Bowl3', 'Table'), (4, 'Bowl2', None), (5, 'Bowl2', 'Bowl6'), \
    #         (6, 'Bowl2', 'Table'), (7, 'Box2','Box3'), (8, 'Bowl5', None), (9, 'Bowl5', 'Bowl6'), (10, 'Bowl5', 'Table'), (11, 'Bowl4', None),\
    #        (12, 'Bowl4', 'Bowl6'), (13, 'Bowl4', 'Table'), (14, 'Bowl1', None), (15,'Bowl1','Bowl6'), (16, 'Bowl1', 'Table') , (17, 'Bowl6', 'Bowl7')]

    # mix_main(mix)
    
    # # ################################################### [Mixing_v2 / 12_3_4_5] ########################################################
    # make_data = MakeDataset(problem = 'mixing_v2', example= '12_3_4_5') 
    
    # mix = [(0, None, None), (1, 'Bowl2', None), (2, 'Bowl2', 'Bowl6'), (3, 'Bowl2', 'Table'), (4, 'Bowl1', None), (5, 'Bowl1', 'Bowl6'), \
    #         (6, 'Bowl1', 'Table'), (7, 'Box1','Box2'), (8, 'Bowl5', None), (9, 'Bowl5', 'Bowl6'), (10, 'Bowl5', 'Table'), (11, 'Bowl4', None),\
    #        (12, 'Bowl4', 'Bowl6'), (13, 'Bowl4', 'Table'), (14, 'Bowl3', None), (15,'Bowl3','Bowl6'), (16, 'Bowl3', 'Table') , (17, 'Bowl6', 'Bowl7')]

    # mix_main(mix)
   

    # #  ################################################## [Mixing_v3 / 1_2_345] ########################################################
    # make_data = MakeDataset(problem = 'mixing_v3', example= '1_2_345')
    # mix = [(0, None, None), (1, 'Bowl5', None), (2, 'Bowl5', 'Bowl6'), (3, 'Bowl5', 'Table'), (4, 'Bowl4', None), (5, 'Bowl4', 'Bowl6'), \
    #     (6, 'Bowl4', 'Table'), (7, 'Box4','Box5'), (8, 'Bowl3', None), (9, 'Bowl3', 'Bowl6'), (10, 'Bowl3', 'Table'), (11, 'Box3', 'Box4 + Box5'), \
    #     (12, 'Bowl2', None), (13, 'Bowl2', 'Bowl6'), (14, 'Bowl2', 'Table'), (15, 'Bowl1', None), (16,'Bowl1','Bowl6'), (17, 'Bowl1', 'Table') , (18, 'Bowl6', 'Bowl7')]
    # mix_main(mix)
    
    # #  ################################################## [Mixing_v3 / 1_234_5] ########################################################
    # make_data = MakeDataset(problem = 'mixing_v3', example= '1_234_5')
    # mix = [(0, None, None), (1, 'Bowl3', None), (2, 'Bowl3', 'Bowl6'), (3, 'Bowl3', 'Table'), (4, 'Bowl4', None), (5, 'Bowl4', 'Bowl6'), \
    #     (6, 'Bowl4', 'Table'), (7, 'Box3','Box4'), (8, 'Bowl2', None), (9, 'Bowl2', 'Bowl6'), (10, 'Bowl2', 'Table'), (11, 'Box2', 'Box3 + Box4'), \
    #     (12, 'Bowl5', None), (13, 'Bowl5', 'Bowl6'), (14, 'Bowl5', 'Table'), (15, 'Bowl1', None), (16,'Bowl1','Bowl6'), (17, 'Bowl1', 'Table') , (18, 'Bowl6', 'Bowl7')]
    # mix_main(mix)

    #  ################################################## [Mixing_v3 / 123_4_5] ########################################################
    # make_data = MakeDataset(problem = 'mixing_v3', example= '123_4_5')
    # mix = [(0, None, None), (1, 'Bowl2', None), (2, 'Bowl2', 'Bowl6'), (3, 'Bowl2', 'Table'), (4, 'Bowl3', None), (5, 'Bowl3', 'Bowl6'), \
    #     (6, 'Bowl3', 'Table'), (7, 'Box2','Box3'), (8, 'Bowl1', None), (9, 'Bowl1', 'Bowl6'), (10, 'Bowl1', 'Table'), (11, 'Box1', 'Box2 + Box3'), \
    #     (12, 'Bowl5', None), (13, 'Bowl5', 'Bowl6'), (14, 'Bowl5', 'Table'), (15, 'Bowl4', None), (16,'Bowl4','Bowl6'), (17, 'Bowl4', 'Table') , (18, 'Bowl6', 'Bowl7')]
    # mix_main(mix)


    # #  ################################################## [Mixing_v4 / 1_2345] ########################################################
    # make_data = MakeDataset(problem = 'mixing_v4', example= '1_2345')
    # mix = [(0, None, None), (1, 'Bowl5', None), (2, 'Bowl5', 'Bowl6'), (3, 'Bowl5', 'Table'), (4, 'Bowl4', None), (5, 'Bowl4', 'Bowl6'), \
    #     (6, 'Bowl4', 'Table'), (7, 'Box4','Box5'), (8, 'Bowl3', None), (9, 'Bowl3', 'Bowl6'), (10, 'Bowl3', 'Table'), (11, 'Box3', 'Box4 + Box5'), \
    #     (12, 'Bowl2', None), (13, 'Bowl2', 'Bowl6'), (14, 'Bowl2', 'Table'), (15, 'Box2', 'Box3 + Box4 + Box5'), \
    #     (16, 'Bowl1', None), (17,'Bowl1','Bowl6'), (18, 'Bowl1', 'Table') , (19, 'Bowl6', 'Bowl7')]
    # mix_main(mix)
    #  ################################################## [Mixing_v4 / 1_2345] ########################################################
    make_data = MakeDataset(problem = 'mixing_v4', example= '1234_5')
    mix = [(0, None, None), (1, 'Bowl4', None), (2, 'Bowl4', 'Bowl6'), (3, 'Bowl4', 'Table'), (4, 'Bowl3', None), (5, 'Bowl3', 'Bowl6'), \
        (6, 'Bowl3', 'Table'), (7, 'Box3','Box4'), (8, 'Bowl2', None), (9, 'Bowl2', 'Bowl6'), (10, 'Bowl2', 'Table'), (11, 'Box2', 'Box3 + Box4'), \
        (12, 'Bowl1', None), (13, 'Bowl1', 'Bowl6'), (14, 'Bowl1', 'Table'), (15, 'Box1', 'Box2 + Box3 + Box4'), \
        (16, 'Bowl5', None), (17,'Bowl5','Bowl6'), (18, 'Bowl5', 'Table') , (19, 'Bowl6', 'Bowl7')]
    mix_main(mix)

print("====[END]====")

            