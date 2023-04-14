from seq_demo import *


# class stacking_problem():
def stack_node_features(list):#node feature
    for i, (action, obj1, obj2) in enumerate(list):
        if action == 0:
            print(make_data.pick_nf(n=i+1, obj1=obj1)) 
        elif action == 1:
            print(make_data.place_nf(n=i+1, obj1=obj1, obj2=obj2))
                
def stack_edge_index(list_inx):
    print(make_data.init_edge_index())
    for i, (action, obj1, obj2) in enumerate(list_inx):            
        if action == 0:
            print(make_data.pick_inx(n=i+1, obj1=obj1))
        elif action == 1:
            print(make_data.place_inx(n=i+1, obj1=obj1, obj2=obj2))


def stack_edge_attr(list):
    print(make_data.init_edge_attr(len(list)))
    for i, (action, obj1, obj2) in enumerate(list):
        if action == 0:
            print(make_data.pick_attr(n=i+1, obj1=obj1)) 
        elif action == 1:
            print(make_data.place_attr(n=i+1, obj1=obj1, obj2=obj2))



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
    print(make_data.init_node_features())
    stack_edge_index(list)
    stack_node_features(list)
    stack_edge_attr(list)
    '''
    num = len(list)
    for i in range(num):
        print(f"====================================================[Task{i}]====================================================")
        node1, node2 = make_node_colors(list,i)
        make_data.check_graph(i, node1, node2)
    print("[[[Graph end]]]")
    '''
########################################################### [Mix] ########################################################################
def mix_node_features(list):
    for i, (action, obj1, obj2) in enumerate(list):
        if action == 0:
            print(make_data.pick_nf(n=i+1, obj1=obj1)) 
        elif action == 1:
            print(make_data.place_nf(n=i+1, obj1=obj1, obj2=obj2))
        elif action == 2:
            print(make_data.pour_nf(n=i+1, obj1=obj1, obj2=obj2))
        # elif action == 3:
        #     print(make_data.mix_nf(n=i+1, obj1=obj1, obj2=obj2))
        
def mix_edge_index(list):
    print(make_data.init_edge_index())
    for i, (action, obj1, obj2) in enumerate(list):
        if action == 0:
            print(make_data.pick_inx(n=i+1, obj1=obj1))
        elif action == 1:
            print(make_data.place_inx(n=i+1, obj1=obj1, obj2= obj2))
        elif action == 2:
            print(make_data.pour_inx(n=i+1, obj1=obj1, obj2 = obj2))
        # elif action == 3:
        #     print(make_data.mix_inx(n=i+1, obj1=obj1, obj2 = obj2))
        else:
            print("Wrong task")


def mix_edge_attr(list):
    print(make_data.init_edge_attr(len(list)))
    for i, (action, obj1, obj2) in enumerate(list):
        if action == 0:
            print(make_data.pick_attr(n=i+1, obj1=obj1)) 
        elif action == 1:
            print(make_data.place_attr(n=i+1, obj1=obj1, obj2=obj2))
        elif action == 2:
            print(make_data.pour_attr(n=i+1, obj1=obj1, obj2 = obj2))
        # elif action == 3:
        #     print(make_data.mix_attr(n=i+1, obj1=obj1, obj2 = obj2))
        else:
            print("Wrong task")
            
def mix_main(list):
    
    print(make_data.init_node_features())
    mix_edge_index(list)
    mix_node_features(list)
    mix_edge_attr(list)
    '''
    num = len(list)
    for i in range(num):
        print(f"====================================================[Task{i}]====================================================")
        node1, node2 = make_node_colors(list,i)
        make_data.check_graph(i, node1, node2)
    print("[[[Graph end]]]")
    '''

if __name__ == '__main__':
    #  ################################################### [Action] ########################################################
    #                                           [Pick: 0, Place: 1, Pour: 2]




    #  ################################################### [Mixing_5] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'mixing_5', example= 'pose_' + str(i+1))

    #     mix = [(0, 'Bowl6', None), (1, 'Bowl6', 'Region_Pour'), (0, 'Bowl7', None), (1, 'Bowl7', 'Region_Pour'),
    #            (0, 'Bowl1', None), (2, 'Bowl1', 'Bowl6'), (1, 'Bowl1', 'Region_Bw1'),
    #            (0, 'Bowl2', None), (2, 'Bowl2', 'Bowl6'), (1, 'Bowl2', 'Region_Bw2'),
    #            (0, 'Bowl3', None), (2, 'Bowl3', 'Bowl6'), (1, 'Bowl3', 'Region_Bw3'),
    #            (0, 'Bowl4', None), (2, 'Bowl4', 'Bowl6'), (1, 'Bowl4', 'Region_Bw4'),
    #            (0, 'Bowl5', None), (2, 'Bowl5', 'Bowl6'), (1, 'Bowl5', 'Region_Bw5'),
    #            (0, 'Bowl6', None), (2, 'Bowl6', 'Bowl7'), (1, 'Bowl6', 'Region_Bw6')]
    #     mix_main(mix)

    #  ################################################### [Mixing_2] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'mixing_2', example= 'pose_' + str(i+1))

    #     mix = [(0, 'Bowl6', None), (1, 'Bowl6', 'Region_Pour'), (0, 'Bowl7', None), (1, 'Bowl7', 'Region_Pour'),
    #            (0, 'Bowl1', None), (2, 'Bowl1', 'Bowl6'), (1, 'Bowl1', 'Region_Bw1'),
    #            (0, 'Bowl2', None), (2, 'Bowl2', 'Bowl6'), (1, 'Bowl2', 'Region_Bw2'),
    #            (0, 'Bowl6', None), (2, 'Bowl6', 'Bowl7'), (1, 'Bowl6', 'Region_Bw6')]
    #     mix_main(mix)
    #  ################################################### [Mixing_3] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'mixing_3', example= 'pose_' + str(i+1))

    #     mix = [(0, 'Bowl6', None), (1, 'Bowl6', 'Region_Pour'), (0, 'Bowl7', None), (1, 'Bowl7', 'Region_Pour'),
    #            (0, 'Bowl1', None), (2, 'Bowl1', 'Bowl6'), (1, 'Bowl1', 'Region_Bw1'),
    #            (0, 'Bowl2', None), (2, 'Bowl2', 'Bowl6'), (1, 'Bowl2', 'Region_Bw2'),
    #            (0, 'Bowl3', None), (2, 'Bowl3', 'Bowl6'), (1, 'Bowl3', 'Region_Bw3'),
    #            (0, 'Bowl6', None), (2, 'Bowl6', 'Bowl7'), (1, 'Bowl6', 'Region_Bw6')]
    #     mix_main(mix)
    #  ################################################### [Mixing_4] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'mixing_4', example= 'pose_' + str(i+1))

    #     mix = [(0, 'Bowl6', None), (1, 'Bowl6', 'Region_Pour'), (0, 'Bowl7', None), (1, 'Bowl7', 'Region_Pour'),
    #            (0, 'Bowl1', None), (2, 'Bowl1', 'Bowl6'), (1, 'Bowl1', 'Region_Bw1'),
    #            (0, 'Bowl2', None), (2, 'Bowl2', 'Bowl6'), (1, 'Bowl2', 'Region_Bw2'),
    #            (0, 'Bowl3', None), (2, 'Bowl3', 'Bowl6'), (1, 'Bowl3', 'Region_Bw3'),
    #            (0, 'Bowl4', None), (2, 'Bowl4', 'Bowl6'), (1, 'Bowl4', 'Region_Bw4'),
    #            (0, 'Bowl6', None), (2, 'Bowl6', 'Bowl7'), (1, 'Bowl6', 'Region_Bw6')]
    #     mix_main(mix)

    #  ################################################### [mixing_withbox2] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'mixing_withbox2', example= 'pose_' + str(i+1))

    #     mix = [(0, 'Bowl6', None), (1, 'Bowl6', 'Region_Pour'),(0, 'Bowl7', None), (1, 'Bowl7', 'Region_Pour'),
    #            (0, 'Box1', None), (1, 'Box1', 'Bowl6'),(0, 'Box2', None), (1, 'Box2', 'Bowl6'),
    #            (0, 'Bowl1', None), (2, 'Bowl1', 'Bowl6'), (1, 'Bowl1', 'Region_Bw1'),
    #            (0, 'Bowl2', None), (2, 'Bowl2', 'Bowl6'), (1, 'Bowl2', 'Region_Bw2'),
    #            (0, 'Bowl6', None), (2, 'Bowl6', 'Bowl7'), (1, 'Bowl6', 'Region_Bw6')]
    #     mix_main(mix)
    #  ################################################### [mixing_withbox3] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'mixing_withbox3', example= 'pose_' + str(i+1))

    #     mix = [(0, 'Bowl6', None), (1, 'Bowl6', 'Region_Pour'),(0, 'Bowl7', None), (1, 'Bowl7', 'Region_Pour'),
    #            (0, 'Box1', None), (1, 'Box1', 'Bowl6'),(0, 'Box2', None), (1, 'Box2', 'Bowl6'),(0, 'Box3', None), (1, 'Box3', 'Bowl6'),
    #            (0, 'Bowl1', None), (2, 'Bowl1', 'Bowl6'), (1, 'Bowl1', 'Region_Bw1'),
    #            (0, 'Bowl2', None), (2, 'Bowl2', 'Bowl6'), (1, 'Bowl2', 'Region_Bw2'),
    #            (0, 'Bowl3', None), (2, 'Bowl3', 'Bowl6'), (1, 'Bowl3', 'Region_Bw3'),
    #            (0, 'Bowl6', None), (2, 'Bowl6', 'Bowl7'), (1, 'Bowl6', 'Region_Bw6')]
    #     mix_main(mix)

    #  ################################################### [mixing_withbox4] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'mixing_withbox4', example= 'pose_' + str(i+1))

    #     mix = [(0, 'Bowl6', None), (1, 'Bowl6', 'Region_Pour'),(0, 'Bowl7', None), (1, 'Bowl7', 'Region_Pour'),
    #            (0, 'Box1', None), (1, 'Box1', 'Bowl6'),(0, 'Box2', None), (1, 'Box2', 'Bowl6'),
    #            (0, 'Box3', None), (1, 'Box3', 'Bowl6'),(0, 'Box4', None), (1, 'Box4', 'Bowl6'),
    #            (0, 'Bowl1', None), (2, 'Bowl1', 'Bowl6'), (1, 'Bowl1', 'Region_Bw1'),
    #            (0, 'Bowl2', None), (2, 'Bowl2', 'Bowl6'), (1, 'Bowl2', 'Region_Bw2'),
    #            (0, 'Bowl3', None), (2, 'Bowl3', 'Bowl6'), (1, 'Bowl3', 'Region_Bw3'),
    #            (0, 'Bowl4', None), (2, 'Bowl4', 'Bowl6'), (1, 'Bowl4', 'Region_Bw4'),
    #            (0, 'Bowl6', None), (2, 'Bowl6', 'Bowl7'), (1, 'Bowl6', 'Region_Bw6')]
    #     mix_main(mix)

    # ################################################### [mixing_withbox5] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'mixing_withbox5', example= 'pose_' + str(i+1))

    #     mix = [(0, 'Bowl6', None), (1, 'Bowl6', 'Region_Pour'),(0, 'Bowl7', None), (1, 'Bowl7', 'Region_Pour'),
    #            (0, 'Box1', None), (1, 'Box1', 'Bowl6'),(0, 'Box2', None), (1, 'Box2', 'Bowl6'),
    #            (0, 'Box3', None), (1, 'Box3', 'Bowl6'),(0, 'Box4', None), (1, 'Box4', 'Bowl6'),(0, 'Box5', None), (1, 'Box5', 'Bowl6'),
    #            (0, 'Bowl1', None), (2, 'Bowl1', 'Bowl6'), (1, 'Bowl1', 'Region_Bw1'),
    #            (0, 'Bowl2', None), (2, 'Bowl2', 'Bowl6'), (1, 'Bowl2', 'Region_Bw2'),
    #            (0, 'Bowl3', None), (2, 'Bowl3', 'Bowl6'), (1, 'Bowl3', 'Region_Bw3'),
    #            (0, 'Bowl4', None), (2, 'Bowl4', 'Bowl6'), (1, 'Bowl4', 'Region_Bw4'),
    #            (0, 'Bowl5', None), (2, 'Bowl5', 'Bowl6'), (1, 'Bowl5', 'Region_Bw5'),
    #            (0, 'Bowl6', None), (2, 'Bowl6', 'Bowl7'), (1, 'Bowl6', 'Region_Bw6')]
    #     mix_main(mix)

    #  ################################################### [Stacking_5] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'stacking_5', example= 'pose_' + str(i+1))

    #     stack = [(0, 'Box4', None), (1, 'Box4', 'Box5'), (0, 'Box3', None), (1, 'Box3', 'Box4'), (0, 'Box2', None), \
    #                     (1, 'Box2', 'Box3'), (0, 'Box1', None), (1, 'Box1', 'Box2')]
    #     stack_main(stack)

    #  ################################################### [Stacking_init2] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'stacking_init2', example= 'pose_' + str(i+1))

    #     stack = [(0, 'Box2', None), (1, 'Box2', 'Box3'), (0, 'Box1', None), (1, 'Box1', 'Box2')]
    #     stack_main(stack)


    #  ################################################### [Stacking_init3] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'stacking_init3', example= 'pose_' + str(i+1))

    #     stack = [(0, 'Box1', None), (1, 'Box1', 'Box2')]
    #     stack_main(stack)


    #  ################################################### [Stacking_init3_replace] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'stacking_init3_replace', example= 'pose_' + str(i+1))

    #     stack = [(0, 'Box3', None), (1, 'Box3', 'Region_Free'), (0, 'Box4', None), (1, 'Box4', 'Region_Free'), \
    #              (0, 'Box2', None), (1, 'Box2', 'Box5'), (0, 'Box3', None), (1, 'Box3', 'Box2')]
    #     stack_main(stack)
    

    #  ################################################### [Stacking_init3_reverse] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'stacking_init3_reverse', example= 'pose_' + str(i+1))

    #     stack = [(0, 'Box3', None), (1, 'Box3', 'Region_Free'), (0, 'Box4', None), (1, 'Box4', 'Region_Free'), \
    #              (0, 'Box5', None), (1, 'Box5', 'Region_Free'), (0, 'Box4', None), (1, 'Box4', 'Box3'), \
    #              (0, 'Box5', None), (1, 'Box5', 'Box4')]
    #     stack_main(stack)

    #  ################################################### [cleaning_box] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'cleaning_box', example= 'pose_' + str(i+1))

    #     stack = [(0, 'Box1', None), (1, 'Box1', 'Region_Clean'), (0, 'Box2', None), (1, 'Box2', 'Region_Clean'),
    #              (0, 'Box3', None), (1, 'Box3', 'Region_Clean'), (0, 'Box4', None), (1, 'Box4', 'Region_Clean'),
    #              (0, 'Box5', None), (1, 'Box5', 'Region_Clean'), (0, 'Box6', None), (1, 'Box6', 'Region_Clean'),
    #              (0, 'Box7', None), (1, 'Box7', 'Region_Clean'), (0, 'Box8', None), (1, 'Box8', 'Region_Clean'),
    #              (0, 'Box9', None), (1, 'Box9', 'Region_Clean')]
    #     stack_main(stack)

    #  ################################################### [cleaning_init5] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'cleaning_init5', example= 'pose_' + str(i+1))

    #     stack = [(0, 'Box6', None), (1, 'Box6', 'Region_Clean'),(0, 'Box7', None), (1, 'Box7', 'Region_Clean'),
    #              (0, 'Box8', None), (1, 'Box8', 'Region_Clean'),(0, 'Box9', None), (1, 'Box9', 'Region_Clean')]
    #     stack_main(stack)
    #  ################################################### [cleaning_init4] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'cleaning_init4', example= 'pose_' + str(i+1))

    #     stack = [(0, 'Box6', None), (1, 'Box6', 'Region_Clean'),(0, 'Box7', None), (1, 'Box7', 'Region_Clean'),
    #              (0, 'Box8', None), (1, 'Box8', 'Region_Clean'),(0, 'Box9', None), (1, 'Box9', 'Region_Clean'),
    #              (0, 'Box1', None), (1, 'Box1', 'Region_Clean')]
    #     stack_main(stack)
    #  ################################################### [cleaning_init3] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'cleaning_init3', example= 'pose_' + str(i+1))

    #     stack = [(0, 'Box6', None), (1, 'Box6', 'Region_Clean'),(0, 'Box7', None), (1, 'Box7', 'Region_Clean'),
    #              (0, 'Box8', None), (1, 'Box8', 'Region_Clean'),(0, 'Box9', None), (1, 'Box9', 'Region_Clean'),
    #              (0, 'Box1', None), (1, 'Box1', 'Region_Clean'),(0, 'Box2', None), (1, 'Box2', 'Region_Clean')]
    #     stack_main(stack)
    #  ################################################### [cleaning_init2] ########################################################
    # for i in range(5):
    #     make_data = MakeDataset(problem = 'cleaning_init2', example= 'pose_' + str(i+1))

    #     stack = [(0, 'Box6', None), (1, 'Box6', 'Region_Clean'),(0, 'Box7', None), (1, 'Box7', 'Region_Clean'),
    #              (0, 'Box8', None), (1, 'Box8', 'Region_Clean'),(0, 'Box9', None), (1, 'Box9', 'Region_Clean'),
    #              (0, 'Box1', None), (1, 'Box1', 'Region_Clean'),(0, 'Box2', None), (1, 'Box2', 'Region_Clean'),
    #              (0, 'Box3', None), (1, 'Box3', 'Region_Clean')]
    #     stack_main(stack)


print("====[END]====")

            