import numpy as np
import os
import sys
import torch
from math import pi



'''
class Stack5PoseGenerator():
    def __init__(self):
        super(Stack5PoseGenerator, self).__init__()
        self.box_size = 0.04
        self.bowl_size = 0.16
        self.franka_h = 0.375
        self.table_h = 0.18
        self.in_grasp = None
        self.current_scene = None
        self.obj_dict = {"Robot_hand": [0, 0],
                         "Box1": [1, self.box_size],
                         "Box2": [2, self.box_size],
                         "Box3": [3, self.box_size],
                         "Box4": [4, self.box_size],
                         "Box5": [5, self.box_size],
                         "Bowl1": [6, self.bowl_size],
                         "Bowl2": [7, self.bowl_size],
                         "Table": [8, 0]}
        self.pose_dir = None

    def random_translation(self):
        translation = np.concatenate([np.random.uniform(-0.15, 0.15, 1),np.random.uniform(-0.025, 0.025, 1), np.array([0])], 0)
        return translation


    def random_orientation(self):
        orientation = np.append(np.array([0, 0]), np.random.uniform(-pi/4, pi/4, 1))
        return orientation

    def random_box_generation(self):
        #generate random poses for 5 boxes
        box_order = [-0.2,-0.1,0,0.1,0.2]
        np.random.shuffle(box_order)

        pose_list = []
        for y_center in box_order:
            translation = np.array([0, y_center, self.box_size/2]) + self.random_translation()
            orientation = self.random_orientation()
            pose_list.append(np.concatenate([translation, orientation], 0))
        box_poses = np.stack(pose_list, axis=0)
        return box_poses

    def bowl_generation(self):
        bowl1 = np.array([0, 0.5, 0, 0, 0, 0])
        bowl2 = np.array([0, -0.5, 0, 0, 0, 0])
        return np.stack([bowl1, bowl2], axis=0)

    def initial_scene(self):
        #coordinate center = table center point
        #robot hand pose : 임의로 설정함
        robot_hand = np.array([[0, 0, 0.5, 0, 0, 0]])
        box_poses = self.random_box_generation()
        bowl_poses = self.bowl_generation()
        table_pose = np.array([[0, 0, 0, 0, 0, 0]])
        init_scene = np.concatenate([robot_hand, box_poses, bowl_poses, table_pose], axis=0)
        return init_scene

    def Pick(self,obj):
        obj_idx = self.obj_dict[obj][0]
        scene = self.current_scene
        scene[obj_idx, :] = scene[0, :]
        self.in_grasp = obj
        return scene
    
    def Place(self, obj):
        obj_idx = self.obj_dict[obj][0]
        in_grasp_idx = self.obj_dict[self.in_grasp][0]
        scene = self.current_scene
        scene[in_grasp_idx,:] = scene[obj_idx,:]
        scene[in_grasp_idx,2] += (self.obj_dict[self.in_grasp][1] + self.obj_dict[obj][1])/2
        self.in_grasp = None
        return scene
    
    def Pour(self, obj):
        pass
    def Mix(self, obj):
        pass
    def hierarchical_pose_update(self, obj1, obj2):
        #mean
        new_obj = obj1 + " + " + obj2
        scene = self.current_scene
        scene[self.obj_dict[new_obj][0]] = np.mean([scene[self.obj_dict[obj1][0]],
                                                    scene[self.obj_dict[obj2][0]]], axis=0)
        return scene
    def pose_generate(self):
        self.pose_dir_init()
        
        self.current_scene = self.initial_scene()
        #save
        self.save_data(0)

        #Pick Box4
        self.current_scene = self.Pick("Box4")
        #save
        self.save_data(1)

        #Place Box5
        self.current_scene = self.Place("Box5")
        #save
        self.save_data(2)

        #Pick Box3
        self.current_scene = self.Pick("Box3")
        #save
        self.save_data(3)

        #Place Box4
        self.current_scene = self.Place("Box4")
        #save
        self.save_data(4)

        #Pick Box2
        self.current_scene = self.Pick("Box2")
        #save
        self.save_data(5)

        #Place Box3
        self.current_scene = self.Place("Box3")
        #save
        self.save_data(6)

        #Pick Box1
        self.current_scene = self.Pick("Box1")
        #save
        self.save_data(7)

        #Place Box2
        self.current_scene = self.Place("Box2")
        #save
        self.save_data(8)


    def pose_dir_init(self):
        #액션 다 짜고나서
        data_dir = '/home/byeon/cooking_branches/byeon_8/cooking/seq_dataset/tasks/stacking_5/1_2_3_4_5/pose/demo_'
        demo_idx = 0
        while os.path.exists(data_dir + str(demo_idx)):
            demo_idx += 1
        data_dir = data_dir + str(demo_idx)
        os.makedirs(data_dir)
        self.pose_dir = data_dir
    
    def save_data(self, step):
        #f = open(self.pose_dir + '/pose_'+str(step)+'.csv', 'w')
        #np.savetxt(self.pose_dir + '/pose_'+str(step)+'.csv', np.round(self.current_scene, 5), delimiter=',')
        np.savetxt(self.pose_dir + '/pose_'+str(step)+'.csv', self.current_scene, delimiter=',')
        
PoseGenerator = Stack5PoseGenerator()
for i in range(100):
    PoseGenerator.pose_generate()
'''
'''
class StackV2PoseGenerator():
    def __init__(self):
        super(StackV2PoseGenerator, self).__init__()
        self.box_size = 0.04
        self.bowl_size = 0.16
        self.franka_h = 0.375
        self.table_h = 0.18
        self.in_grasp = None
        self.current_scene = None
        self.obj_dict = {"Robot_hand": [0, 0],
                         "Box1": [1, self.box_size],
                         "Box2": [2, self.box_size],
                         "Box3": [3, self.box_size],
                         "Box4": [4, self.box_size],
                         "Box5": [5, self.box_size],
                         "Bowl1": [6, self.bowl_size],
                         "Bowl2": [7, self.bowl_size],
                         "Table": [8, 0],}
        self.order_seq_dict = {'1_2_3_45':["Box4", "Box5", "Box3", "Box4 + Box5", "Box2", "Box3", "Box1", "Box2"],
                               '1_2_34_5':["Box3", "Box4", "Box3 + Box4", "Box5", "Box2", "Box3 + Box4", "Box1", "Box2"],
                               '1_23_4_5':["Box4", "Box5", "Box2", "Box3", "Box2 + Box3", "Box4", "Box1", "Box2 + Box3"],
                               '12_3_4_5':["Box4", "Box5", "Box3", "Box4",  "Box1", "Box2", "Box1 + Box2", "Box3"]}
        self.pose_dir = None

    def random_translation(self):
        translation = np.concatenate([np.random.uniform(-0.15, 0.15, 1),np.random.uniform(-0.025, 0.025, 1), np.array([0])], 0)
        return translation


    def random_orientation(self):
        orientation = np.append(np.array([0, 0]), np.random.uniform(-pi/4, pi/4, 1))
        return orientation

    def random_box_generation(self):
        #generate random poses for 5 boxes
        box_order = [-0.2,-0.1,0,0.1,0.2]
        np.random.shuffle(box_order)

        pose_list = []
        for y_center in box_order:
            translation = np.array([0, y_center, self.box_size/2]) + self.random_translation()
            orientation = self.random_orientation()
            pose_list.append(np.concatenate([translation, orientation], 0))
        box_poses = np.stack(pose_list, axis=0)
        return box_poses

    def bowl_generation(self):
        bowl1 = np.array([0, 0.5, 0, 0, 0, 0])
        bowl2 = np.array([0, -0.5, 0, 0, 0, 0])
        return np.stack([bowl1, bowl2], axis=0)

    def initial_scene(self):
        #coordinate center = table center point
        #robot hand pose : 임의로 설정함
        robot_hand = np.array([[0, 0, 0.5, 0, 0, 0]])
        box_poses = self.random_box_generation()
        bowl_poses = self.bowl_generation()
        table_pose = np.array([[0, 0, 0, 0, 0, 0]])
        zeros_pose = np.array([[0, 0, 0, 0, 0, 0]])
        init_scene = np.concatenate([robot_hand, box_poses, bowl_poses, table_pose, zeros_pose], axis=0)
        return init_scene

    def Pick(self,obj):
        obj_idx = self.obj_dict[obj][0]
        scene = self.current_scene
        scene[obj_idx, :] = scene[0, :]
        self.in_grasp = obj
        return scene
    
    def Place(self, obj):
        obj_idx = self.obj_dict[obj][0]
        in_grasp_idx = self.obj_dict[self.in_grasp][0]
        scene = self.current_scene
        scene[in_grasp_idx,:] = scene[obj_idx,:]
        scene[in_grasp_idx,2] += (self.obj_dict[self.in_grasp][1] + self.obj_dict[obj][1])/2
        self.in_grasp = None
        return scene
    
    def Pour(self, obj):
        pass
    def Mix(self, obj):
        pass
    def hierarchical_pose_update(self, obj1, obj2):
        #mean
        new_obj = obj1 + " + " + obj2
        scene = self.current_scene
        scene[self.obj_dict[new_obj][0]] = np.mean([scene[self.obj_dict[obj1][0]],
                                                    scene[self.obj_dict[obj2][0]]], axis=0)
        return scene
    def pose_generate(self, order):
        self.pose_dir_init(order)
        self.obj_sequence = self.order_seq_dict[order]
        
        self.current_scene = self.initial_scene()
        #save
        self.save_data(0)

        #Pick Box4
        self.current_scene = self.Pick(self.obj_sequence[0])
        #save
        self.save_data(1)

        #Place Box5
        self.current_scene = self.Place(self.obj_sequence[1])
        if (order == "1_2_3_45") or (order == "1_2_34_5"):
            new_obj = self.obj_sequence[0] + " + " + self.obj_sequence[1]
            self.obj_dict[new_obj] = [9, self.obj_dict[self.obj_sequence[0]][1] + self.obj_dict[self.obj_sequence[1]][1]]
            self.current_scene = self.hierarchical_pose_update(self.obj_sequence[0], self.obj_sequence[1])
        #save
        self.save_data(2)

        #Pick Box3
        self.current_scene = self.Pick(self.obj_sequence[2])
        #save
        self.save_data(3)

        #Place Box4
        self.current_scene = self.Place(self.obj_sequence[3])
        if order == "1_23_4_5":
            new_obj = self.obj_sequence[2] + " + " + self.obj_sequence[3]
            self.obj_dict[new_obj] = [9, self.obj_dict[self.obj_sequence[2]][1] + self.obj_dict[self.obj_sequence[3]][1]]
            self.current_scene = self.hierarchical_pose_update(self.obj_sequence[2], self.obj_sequence[3])
        #save
        self.save_data(4)

        #Pick Box2
        self.current_scene = self.Pick(self.obj_sequence[4])
        #save
        self.save_data(5)

        #Place Box3
        self.current_scene = self.Place(self.obj_sequence[5])
        if order == "12_3_4_5":
            new_obj = self.obj_sequence[4] + " + " + self.obj_sequence[5]
            self.obj_dict[new_obj] = [9, self.obj_dict[self.obj_sequence[4]][1] + self.obj_dict[self.obj_sequence[5]][1]]
            self.current_scene = self.hierarchical_pose_update(self.obj_sequence[4], self.obj_sequence[5])
        #save
        self.save_data(6)

        #Pick Box1
        self.current_scene = self.Pick(self.obj_sequence[6])
        #save
        self.save_data(7)

        #Place Box2
        self.current_scene = self.Place(self.obj_sequence[7])
        #save
        self.save_data(8)


    def pose_dir_init(self, order):
        #액션 다 짜고나서
        data_dir = '/home/byeon/cooking_branches/byeon_8/cooking/seq_dataset/tasks/stacking_v2/'+ order+ '/pose/demo_'
        demo_idx = 0
        while os.path.exists(data_dir + str(demo_idx)):
            demo_idx += 1
        data_dir = data_dir + str(demo_idx)
        os.makedirs(data_dir)
        self.pose_dir = data_dir
    
    def save_data(self, step):
        #f = open(self.pose_dir + '/pose_'+str(step)+'.csv', 'w')
        #np.savetxt(self.pose_dir + '/pose_'+str(step)+'.csv', np.round(self.current_scene, 5), delimiter=',')
        np.savetxt(self.pose_dir + '/pose_'+str(step)+'.csv', self.current_scene, delimiter=',')

PoseGeneratorv2_45 = StackV2PoseGenerator()
for i in range(100):
    PoseGeneratorv2_45.pose_generate('1_2_3_45')
PoseGeneratorv2_34 = StackV2PoseGenerator()
for i in range(100):   
    PoseGeneratorv2_34.pose_generate('1_2_34_5')
PoseGeneratorv2_23 = StackV2PoseGenerator()
for i in range(100):
    PoseGeneratorv2_23.pose_generate('1_23_4_5')
PoseGeneratorv2_12 = StackV2PoseGenerator()
for i in range(100):
    PoseGeneratorv2_12.pose_generate('12_3_4_5')
'''
'''
class StackV3PoseGenerator():
    def __init__(self):
        super(StackV3PoseGenerator, self).__init__()
        self.box_size = 0.04
        self.bowl_size = 0.16
        self.franka_h = 0.375
        self.table_h = 0.18
        self.in_grasp = None
        self.current_scene = None
        self.obj_dict = {"Robot_hand": [0, 0],
                         "Box1": [1, self.box_size],
                         "Box2": [2, self.box_size],
                         "Box3": [3, self.box_size],
                         "Box4": [4, self.box_size],
                         "Box5": [5, self.box_size],
                         "Bowl1": [6, self.bowl_size],
                         "Bowl2": [7, self.bowl_size],
                         "Table": [8, 0],}
        self.order_seq_dict = {'1_2_345':["Box4", "Box5", "Box3", "Box4 + Box5", "Box2", "Box3 + Box4 + Box5", "Box1", "Box2"],
                               '1_234_5':["Box3", "Box4", "Box2", "Box3 + Box4", "Box2 + Box3 + Box4", "Box5", "Box1", "Box2 + Box3 + Box4"],
                               '123_4_5':["Box4", "Box5", "Box2", "Box3", "Box1", "Box2 + Box3", "Box1 + Box2 + Box3", "Box4"]}
        self.pose_dir = None

    def random_translation(self):
        translation = np.concatenate([np.random.uniform(-0.15, 0.15, 1),np.random.uniform(-0.025, 0.025, 1), np.array([0])], 0)
        return translation


    def random_orientation(self):
        orientation = np.append(np.array([0, 0]), np.random.uniform(-pi/4, pi/4, 1))
        return orientation

    def random_box_generation(self):
        #generate random poses for 5 boxes
        box_order = [-0.2,-0.1,0,0.1,0.2]
        np.random.shuffle(box_order)

        pose_list = []
        for y_center in box_order:
            translation = np.array([0, y_center, self.box_size/2]) + self.random_translation()
            orientation = self.random_orientation()
            pose_list.append(np.concatenate([translation, orientation], 0))
        box_poses = np.stack(pose_list, axis=0)
        return box_poses

    def bowl_generation(self):
        bowl1 = np.array([0, 0.5, 0, 0, 0, 0])
        bowl2 = np.array([0, -0.5, 0, 0, 0, 0])
        return np.stack([bowl1, bowl2], axis=0)

    def initial_scene(self):
        #coordinate center = table center point
        #robot hand pose : 임의로 설정함
        robot_hand = np.array([[0, 0, 0.5, 0, 0, 0]])
        box_poses = self.random_box_generation()
        bowl_poses = self.bowl_generation()
        table_pose = np.array([[0, 0, 0, 0, 0, 0]])
        zeros_pose = np.array([[0, 0, 0, 0, 0, 0]])
        init_scene = np.concatenate([robot_hand, box_poses, bowl_poses, table_pose, zeros_pose, zeros_pose, zeros_pose], axis=0)
        return init_scene

    def Pick(self,obj):
        obj_idx = self.obj_dict[obj][0]
        scene = self.current_scene
        scene[obj_idx, :] = scene[0, :]
        self.in_grasp = obj
        return scene
    
    def Place(self, obj):
        obj_idx = self.obj_dict[obj][0]
        in_grasp_idx = self.obj_dict[self.in_grasp][0]
        scene = self.current_scene
        scene[in_grasp_idx,:] = scene[obj_idx,:]
        scene[in_grasp_idx,2] += (self.obj_dict[self.in_grasp][1] + self.obj_dict[obj][1])/2
        self.in_grasp = None
        return scene
    
    def Pour(self, obj):
        pass
    def Mix(self, obj):
        pass
    def hierarchical_pose_update(self, obj1, obj2):
        #mean
        new_obj = obj1 + " + " + obj2
        scene = self.current_scene
        scene[self.obj_dict[new_obj][0]] = np.mean([scene[self.obj_dict[obj1][0]],
                                                    scene[self.obj_dict[obj2][0]]], axis=0)
        return scene
    def pose_generate(self, order):
        self.pose_dir_init(order)
        self.obj_sequence = self.order_seq_dict[order]
        
        self.current_scene = self.initial_scene()
        #save
        self.save_data(0)

        #Pick Box4
        self.current_scene = self.Pick(self.obj_sequence[0])
        #save
        self.save_data(1)

        #Place Box5
        self.current_scene = self.Place(self.obj_sequence[1])
        if (order == "1_2_345") or (order == "1_234_5"):
            new_obj = self.obj_sequence[0] + " + " + self.obj_sequence[1]
            self.obj_dict[new_obj] = [9, self.obj_dict[self.obj_sequence[0]][1] + self.obj_dict[self.obj_sequence[1]][1]]
            self.current_scene = self.hierarchical_pose_update(self.obj_sequence[0], self.obj_sequence[1])
        #save
        self.save_data(2)

        #Pick Box3
        self.current_scene = self.Pick(self.obj_sequence[2])
        #save
        self.save_data(3)

        #Place Box4
        self.current_scene = self.Place(self.obj_sequence[3])
        if (order == "1_2_345") or (order == "1_234_5"):
            new_obj = self.obj_sequence[2] + " + " + self.obj_sequence[3]
            self.obj_dict[new_obj] = [10, self.obj_dict[self.obj_sequence[2]][1] + self.obj_dict[self.obj_sequence[3]][1]]
            self.current_scene = self.hierarchical_pose_update(self.obj_sequence[2], self.obj_sequence[3])
        else:
            new_obj = self.obj_sequence[2] + " + " + self.obj_sequence[3]
            self.obj_dict[new_obj] = [9, self.obj_dict[self.obj_sequence[2]][1] + self.obj_dict[self.obj_sequence[3]][1]]
            self.current_scene = self.hierarchical_pose_update(self.obj_sequence[2], self.obj_sequence[3])
        #save
        self.save_data(4)

        #Pick Box2
        self.current_scene = self.Pick(self.obj_sequence[4])
        #save
        self.save_data(5)

        #Place Box3
        self.current_scene = self.Place(self.obj_sequence[5])
        if order == "123_4_5":
            new_obj = self.obj_sequence[4] + " + " + self.obj_sequence[5]
            self.obj_dict[new_obj] = [10, self.obj_dict[self.obj_sequence[4]][1] + self.obj_dict[self.obj_sequence[5]][1]]
            self.current_scene = self.hierarchical_pose_update(self.obj_sequence[4], self.obj_sequence[5])
        #save
        self.save_data(6)

        #Pick Box1
        self.current_scene = self.Pick(self.obj_sequence[6])
        #save
        self.save_data(7)

        #Place Box2
        self.current_scene = self.Place(self.obj_sequence[7])
        #save
        self.save_data(8)


    def pose_dir_init(self, order):
        #액션 다 짜고나서
        data_dir = '/home/byeon/cooking_branches/byeon_8/cooking/seq_dataset/tasks/stacking_v3/'+ order+ '/pose/demo_'
        demo_idx = 0
        while os.path.exists(data_dir + str(demo_idx)):
            demo_idx += 1
        data_dir = data_dir + str(demo_idx)
        os.makedirs(data_dir)
        self.pose_dir = data_dir
    
    def save_data(self, step):
        #f = open(self.pose_dir + '/pose_'+str(step)+'.csv', 'w')
        #np.savetxt(self.pose_dir + '/pose_'+str(step)+'.csv', np.round(self.current_scene, 5), delimiter=',')
        np.savetxt(self.pose_dir + '/pose_'+str(step)+'.csv', self.current_scene, delimiter=',')

PoseGeneratorv3_345 = StackV3PoseGenerator()
for i in range(100):
    PoseGeneratorv3_345.pose_generate('1_2_345')
PoseGeneratorv3_234 = StackV3PoseGenerator()
for i in range(100):   
    PoseGeneratorv3_234.pose_generate('1_234_5')
PoseGeneratorv3_123 = StackV3PoseGenerator()
for i in range(100):
    PoseGeneratorv3_123.pose_generate('123_4_5')
'''
'''
class StackV4PoseGenerator():
    def __init__(self):
        super(StackV4PoseGenerator, self).__init__()
        self.box_size = 0.04
        self.bowl_size = 0.16
        self.franka_h = 0.375
        self.table_h = 0.18
        self.in_grasp = None
        self.current_scene = None
        self.obj_dict = {"Robot_hand": [0, 0],
                         "Box1": [1, self.box_size],
                         "Box2": [2, self.box_size],
                         "Box3": [3, self.box_size],
                         "Box4": [4, self.box_size],
                         "Box5": [5, self.box_size],
                         "Bowl1": [6, self.bowl_size],
                         "Bowl2": [7, self.bowl_size],
                         "Table": [8, 0],}
        self.order_seq_dict = {'1_2345':["Box4", "Box5", "Box3", "Box4 + Box5", "Box2", "Box3 + Box4 + Box5", "Box1", "Box2 + Box3 + Box4 + Box5"],
                               '1234_5':["Box3", "Box4", "Box2", "Box3 + Box4", "Box1", "Box2 + Box3 + Box4", "Box1 + Box2 + Box3 + Box4", "Box5"],
                               }
        self.pose_dir = None

    def random_translation(self):
        translation = np.concatenate([np.random.uniform(-0.15, 0.15, 1),np.random.uniform(-0.025, 0.025, 1), np.array([0])], 0)
        return translation


    def random_orientation(self):
        orientation = np.append(np.array([0, 0]), np.random.uniform(-pi/4, pi/4, 1))
        return orientation

    def random_box_generation(self):
        #generate random poses for 5 boxes
        box_order = [-0.2,-0.1,0,0.1,0.2]
        np.random.shuffle(box_order)

        pose_list = []
        for y_center in box_order:
            translation = np.array([0, y_center, self.box_size/2]) + self.random_translation()
            orientation = self.random_orientation()
            pose_list.append(np.concatenate([translation, orientation], 0))
        box_poses = np.stack(pose_list, axis=0)
        return box_poses

    def bowl_generation(self):
        bowl1 = np.array([0, 0.5, 0, 0, 0, 0])
        bowl2 = np.array([0, -0.5, 0, 0, 0, 0])
        return np.stack([bowl1, bowl2], axis=0)

    def initial_scene(self):
        #coordinate center = table center point
        #robot hand pose : 임의로 설정함
        robot_hand = np.array([[0, 0, 0.5, 0, 0, 0]])
        box_poses = self.random_box_generation()
        bowl_poses = self.bowl_generation()
        table_pose = np.array([[0, 0, 0, 0, 0, 0]])
        zeros_pose = np.array([[0, 0, 0, 0, 0, 0]])
        init_scene = np.concatenate([robot_hand, box_poses, bowl_poses, table_pose, zeros_pose, zeros_pose, zeros_pose], axis=0)
        return init_scene

    def Pick(self,obj):
        obj_idx = self.obj_dict[obj][0]
        scene = self.current_scene
        scene[obj_idx, :] = scene[0, :]
        self.in_grasp = obj
        return scene
    
    def Place(self, obj):
        obj_idx = self.obj_dict[obj][0]
        in_grasp_idx = self.obj_dict[self.in_grasp][0]
        scene = self.current_scene
        scene[in_grasp_idx,:] = scene[obj_idx,:]
        scene[in_grasp_idx,2] += (self.obj_dict[self.in_grasp][1] + self.obj_dict[obj][1])/2
        self.in_grasp = None
        return scene
    
    def Pour(self, obj):
        pass
    def Mix(self, obj):
        pass
    def hierarchical_pose_update(self, obj1, obj2):
        #mean
        new_obj = obj1 + " + " + obj2
        scene = self.current_scene
        scene[self.obj_dict[new_obj][0]] = np.mean([scene[self.obj_dict[obj1][0]],
                                                    scene[self.obj_dict[obj2][0]]], axis=0)
        return scene
    def pose_generate(self, order):
        self.pose_dir_init(order)
        self.obj_sequence = self.order_seq_dict[order]
        
        self.current_scene = self.initial_scene()
        #save
        self.save_data(0)

        #Pick Box4
        self.current_scene = self.Pick(self.obj_sequence[0])
        #save
        self.save_data(1)

        #Place Box5
        self.current_scene = self.Place(self.obj_sequence[1])
        new_obj = self.obj_sequence[0] + " + " + self.obj_sequence[1]
        self.obj_dict[new_obj] = [9, self.obj_dict[self.obj_sequence[0]][1] + self.obj_dict[self.obj_sequence[1]][1]]
        self.current_scene = self.hierarchical_pose_update(self.obj_sequence[0], self.obj_sequence[1])
        #save
        self.save_data(2)

        #Pick Box3
        self.current_scene = self.Pick(self.obj_sequence[2])
        #save
        self.save_data(3)

        #Place Box4
        self.current_scene = self.Place(self.obj_sequence[3])
        new_obj = self.obj_sequence[2] + " + " + self.obj_sequence[3]
        self.obj_dict[new_obj] = [10, self.obj_dict[self.obj_sequence[2]][1] + self.obj_dict[self.obj_sequence[3]][1]]
        self.current_scene = self.hierarchical_pose_update(self.obj_sequence[2], self.obj_sequence[3])
        #save
        self.save_data(4)

        #Pick Box2
        self.current_scene = self.Pick(self.obj_sequence[4])
        #save
        self.save_data(5)

        #Place Box3
        self.current_scene = self.Place(self.obj_sequence[5])
        new_obj = self.obj_sequence[4] + " + " + self.obj_sequence[5]
        self.obj_dict[new_obj] = [11, self.obj_dict[self.obj_sequence[4]][1] + self.obj_dict[self.obj_sequence[5]][1]]
        self.current_scene = self.hierarchical_pose_update(self.obj_sequence[4], self.obj_sequence[5])
        #save
        self.save_data(6)

        #Pick Box1
        self.current_scene = self.Pick(self.obj_sequence[6])
        #save
        self.save_data(7)

        #Place Box2
        self.current_scene = self.Place(self.obj_sequence[7])
        #save
        self.save_data(8)


    def pose_dir_init(self, order):
        #액션 다 짜고나서
        data_dir = '/home/byeon/cooking_branches/byeon_8/cooking/seq_dataset/tasks/stacking_v4/'+ order+ '/pose/demo_'
        demo_idx = 0
        while os.path.exists(data_dir + str(demo_idx)):
            demo_idx += 1
        data_dir = data_dir + str(demo_idx)
        os.makedirs(data_dir)
        self.pose_dir = data_dir
    
    def save_data(self, step):
        #f = open(self.pose_dir + '/pose_'+str(step)+'.csv', 'w')
        #np.savetxt(self.pose_dir + '/pose_'+str(step)+'.csv', np.round(self.current_scene, 5), delimiter=',')
        np.savetxt(self.pose_dir + '/pose_'+str(step)+'.csv', self.current_scene, delimiter=',')

PoseGeneratorv4_2345 = StackV4PoseGenerator()
for i in range(100):
    PoseGeneratorv4_2345.pose_generate('1_2345')
PoseGeneratorv4_1234 = StackV4PoseGenerator()
for i in range(100):   
    PoseGeneratorv4_1234.pose_generate('1234_5')

'''

class MixingPoseGenerator():
    def __init__(self):
        super(MixingPoseGenerator, self).__init__()
        self.box_size = 0.04
        self.bowl_size = 0.16
        self.franka_h = 0.375
        self.table_h = 0.18
        self.in_grasp = None
        self.current_scene = None
        self.obj_dict = self.init_obj_dict()
        self.order_seq_dict = {'1_2_3_4_5':["Bowl5", "Bowl6", "Table", "Bowl4", "Bowl6", "Table", "Bowl3", "Bowl6", "Table", "Bowl2", "Bowl6", "Table", "Bowl1", "Bowl6", "Table", "Bowl6", "Bowl7"]}
        self.pose_dir = None

    def random_translation(self):
        translation = np.concatenate([np.random.uniform(-0.15, 0.15, 1),np.random.uniform(-0.025, 0.025, 1), np.array([0])], 0)
        return translation


    def random_orientation(self):
        orientation = np.append(np.array([0, 0]), np.random.uniform(-pi/4, pi/4, 1))
        return orientation

    def random_box_generation(self):
        #generate random poses for 5 boxes
        box_order = [-0.2,-0.1,0,0.1,0.2]
        np.random.shuffle(box_order)

        pose_list = []
        for y_center in box_order:
            translation = np.array([0, y_center, self.box_size/2]) + self.random_translation()
            orientation = self.random_orientation()
            pose_list.append(np.concatenate([translation, orientation], 0))
        box_poses = np.stack(pose_list, axis=0)
        return box_poses

    def bowl_generation(self):
        bowl_left = np.array([0, 0.5, 0, 0, 0, 0])
        bowl_right = np.array([0, -0.5, 0, 0, 0, 0])
        return np.stack([bowl_left, bowl_right], axis=0)

    def initial_scene(self):
        #coordinate center = table center point
        #robot hand pose : 임의로 설정함
        robot_hand = np.array([[0, 0, 0.5, 0, 0, 0]])
        box_poses = self.random_box_generation()
        bowl_poses = box_poses.copy()
        bowl_poses[:, 2] = 0
        side_bowl_poses = self.bowl_generation()
        table_pose = np.array([[0, 0, 0, 0, 0, 0]])
        zeros_pose = np.array([[0, 0, 0, 0, 0, 0]])
        init_scene = np.concatenate([robot_hand, box_poses, bowl_poses, side_bowl_poses, table_pose, zeros_pose, zeros_pose, zeros_pose], axis=0)
        return init_scene

    def Pick(self,obj):
        
        obj_idx = self.obj_dict[obj][0]
        scene = self.current_scene
        scene[obj_idx, :] = scene[0, :]
        self.in_grasp = obj

        in_bowl_idx_list = self.obj_dict[self.in_grasp][2]
        if len(in_bowl_idx_list)>0:
            for in_bowl_idx in in_bowl_idx_list:
                scene[in_bowl_idx,:] = scene[0, :]
        return scene
    
    def Place(self, obj):
        obj_idx = self.obj_dict[obj][0]
        in_grasp_idx = self.obj_dict[self.in_grasp][0]
        scene = self.current_scene
        if obj!='Table':
            scene[in_grasp_idx,:] = scene[obj_idx,:]
            scene[in_grasp_idx,2] += (self.obj_dict[self.in_grasp][1] + self.obj_dict[obj][1])/2
            self.in_grasp = None
            if "Bowl" in obj:
                self.obj_dict[obj][2].append(in_grasp_idx)
        elif obj == 'Table':
            scene[in_grasp_idx,:] = scene[obj_idx,:]
            scene[in_grasp_idx,:3] += np.concatenate([np.random.uniform(-0.15, 0.15, 2), np.array([0])], 0)
            self.in_grasp = None
        return scene
    
    def Pour(self, obj):
        obj_idx = self.obj_dict[obj][0]
        in_bowl_idx_list = self.obj_dict[self.in_grasp][2]
        scene = self.current_scene
        for in_bowl_idx in in_bowl_idx_list:
            scene[in_bowl_idx,:] = scene[obj_idx, :]
            scene[in_bowl_idx, :3] += np.concatenate([np.random.uniform(-self.box_size, self.box_size, 2), np.random.uniform(0, self.bowl_size, 1)], 0)
        self.obj_dict[obj][2] = self.obj_dict[self.in_grasp][2]
        self.obj_dict[self.in_grasp][2] = []
        return scene
    
    def Mix(self, obj):
        pass

    def init_obj_dict(self):
        dict = {"Robot_hand": [0, 0],
                         "Box1": [1, self.box_size],
                         "Box2": [2, self.box_size],
                         "Box3": [3, self.box_size],
                         "Box4": [4, self.box_size],
                         "Box5": [5, self.box_size],
                         "Bowl1": [6, self.bowl_size,[1]],
                         "Bowl2": [7, self.bowl_size,[2]],
                         "Bowl3": [8, self.bowl_size,[3]],
                         "Bowl4": [9, self.bowl_size,[4]],
                         "Bowl5": [10, self.bowl_size,[5]],
                         "Bowl6": [11, self.bowl_size,[]],
                         "Bowl7": [12, self.bowl_size,[]],
                         "Table": [13, 0],}
        return dict
    def hierarchical_pose_update(self, obj1, obj2):
        #mean
        new_obj = obj1 + " + " + obj2
        scene = self.current_scene
        scene[self.obj_dict[new_obj][0]] = np.mean([scene[self.obj_dict[obj1][0]],
                                                    scene[self.obj_dict[obj2][0]]], axis=0)
        return scene
    def pose_generate(self, order):
        self.pose_dir_init(order)
        self.obj_sequence = self.order_seq_dict[order]
        self.obj_dict = self.init_obj_dict()
        self.current_scene = self.initial_scene()
        #save
        self.save_data(0)

        #Pick Bowl
        self.current_scene = self.Pick(self.obj_sequence[0])
        #save
        self.save_data(1)

        #Pour to Bowl
        self.current_scene = self.Pour(self.obj_sequence[1])
        #save
        self.save_data(2)

        #Place Bowl
        self.current_scene = self.Place(self.obj_sequence[2])
        #save
        self.save_data(3)

        #Pick Bowl
        self.current_scene = self.Pick(self.obj_sequence[3])
        #save
        self.save_data(4)

        #Pour to Bowl
        self.current_scene = self.Pour(self.obj_sequence[4])
        #save
        self.save_data(5)

        #Place Bowl
        self.current_scene = self.Place(self.obj_sequence[5])
        #save
        self.save_data(6)

        #Pick Bowl
        self.current_scene = self.Pick(self.obj_sequence[6])
        #save
        self.save_data(7)

        #Pour to Bowl
        self.current_scene = self.Pour(self.obj_sequence[7])
        #save
        self.save_data(8)

        #Place Bowl
        self.current_scene = self.Place(self.obj_sequence[8])
        #save
        self.save_data(9)

        #Pick Bowl
        self.current_scene = self.Pick(self.obj_sequence[9])
        #save
        self.save_data(10)

        #Pour to Bowl
        self.current_scene = self.Pour(self.obj_sequence[10])
        #save
        self.save_data(11)

        #Place Bowl
        self.current_scene = self.Place(self.obj_sequence[11])
        #save
        self.save_data(12)
        #Pick Bowl
        self.current_scene = self.Pick(self.obj_sequence[12])
        #save
        self.save_data(13)

        #Pour to Bowl
        self.current_scene = self.Pour(self.obj_sequence[13])
        #save
        self.save_data(14)

        #Place Bowl
        self.current_scene = self.Place(self.obj_sequence[14])
        #save
        self.save_data(15)

        #Pick Bowl
        self.current_scene = self.Pick(self.obj_sequence[15])
        #save
        self.save_data(16)

        #Pour to Bowl
        self.current_scene = self.Pour(self.obj_sequence[16])
        #save
        self.save_data(17)


    def pose_dir_init(self, order):
        #액션 다 짜고나서
        data_dir = '/home/byeon/cooking_branches/byeon_8/cooking/seq_dataset/tasks/mixing_5/'+ order+ '/pose/demo_'
        demo_idx = 0
        while os.path.exists(data_dir + str(demo_idx)):
            demo_idx += 1
        data_dir = data_dir + str(demo_idx)
        os.makedirs(data_dir)
        self.pose_dir = data_dir
    
    def save_data(self, step):
        #f = open(self.pose_dir + '/pose_'+str(step)+'.csv', 'w')
        #np.savetxt(self.pose_dir + '/pose_'+str(step)+'.csv', np.round(self.current_scene, 5), delimiter=',')
        np.savetxt(self.pose_dir + '/pose_'+str(step)+'.csv', self.current_scene, delimiter=',')

MixPoseGenerator = MixingPoseGenerator()
for i in range(100):
    MixPoseGenerator.pose_generate('1_2_3_4_5')


