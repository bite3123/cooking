import numpy as np
import os
import sys
import torch
from math import pi



class Stack5PoseGenerator():
    def __init__(self):
        super(Stack5PoseGenerator, self).__init__()
        self.box_size = 0.04
        self.franka_h = 0.375
        self.table_h = 0.18
        self.in_grasp = None
        self.current_scene = None
        self.obj_dict = {"Robot_hand": 0,
                         "Box1": 1,
                         "Box2": 2,
                         "Box3": 3,
                         "Box4": 4,
                         "Box5": 5,
                         "Bowl1": 6,
                         "Bowl2": 7,
                         "Table": 8}
        
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
        scene = self.current_scene
        scene[obj, :] = scene[0, :]
        return scene
    
    def Place(self, obj):
        scene = self.current_scene
        scene[self.in_grasp, :] = scene[obj, :]
        scene[self.in_grasp, 2] += self.box_size
        return scene
    
    def Pour(self, obj):
        pass
    def Mix(self, obj):
        pass

    def pose_generate(self):
        self.pose_dir_init()
        
        self.current_scene = self.initial_scene()
        #save
        self.save_data(0)

        #Pick Box4
        self.current_scene = self.Pick(self.obj_dict["Box4"])
        #save
        self.save_data(1)

        #Place Box5
        self.current_scene = self.Place(self.obj_dict["Box5"])
        #save
        self.save_data(2)

        #Pick Box3
        self.current_scene = self.Pick(self.obj_dict["Box3"])
        #save
        self.save_data(3)

        #Place Box4
        self.current_scene = self.Place(self.obj_dict["Box4"])
        #save
        self.save_data(4)

        #Pick Box2
        self.current_scene = self.Pick(self.obj_dict["Box2"])
        #save
        self.save_data(5)

        #Place Box3
        self.current_scene = self.Place(self.obj_dict["Box3"])
        #save
        self.save_data(6)

        #Pick Box1
        self.current_scene = self.Pick(self.obj_dict["Box1"])
        #save
        self.save_data(7)

        #Place Box2
        self.current_scene = self.Place(self.obj_dict["Box2"])
        #save
        self.save_data(8)


    def pose_dir_init(self):
        #액션 다 짜고나서
        data_dir = '/home/byeon/cooking_branches/byeon_8/cooking/seq_dataset/tasks/stacking_5/pose/demo_'
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
for i in range(1000):
    PoseGenerator.pose_generate()