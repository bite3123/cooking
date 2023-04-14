from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import  random_split
import torch.optim as optim
from GraphPlanningDataset import *
from ActionModel import *
import matplotlib.pyplot as plt
import pickle
import os
def inference_action(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num=None, check_each = False):
    
    model_param = [data_dir, hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(), "result", "action","_".join(list(map(str, model_param))))

    if infer_num is not None:
        model_name = 'GP_model_{}.pt'.format(infer_num)
    else:
        model_name = 'GP_model_best.pt'


    with open(os.path.join(model_path, "loss_data"), "rb") as file:
        loss_data = pickle.load(file)
    
    epoch_list = loss_data["epoch"]
    
    #Loss Plot
    plt.figure(0)
    plt.figure(figsize=(15, 5))
    plt.suptitle('Loss')

    #Total Loss
    loss_list = []
    val_loss_list = []
    for loss in loss_data['loss']['total']['train']:
        loss_list.append(loss)
    for val_loss in loss_data['loss']['total']['val']:
        val_loss_list.append(val_loss)
    plt.subplot(1,3,1)
    plt.plot(epoch_list, loss_list, label='train')
    plt.plot(epoch_list, val_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('total')

    #Action Loss
    act_loss_list = []
    val_act_loss_list = []
    for loss in loss_data['loss']['action']['train']:
        act_loss_list.append(loss)
    for val_loss in loss_data['loss']['action']['val']:
        val_act_loss_list.append(val_loss)
    plt.subplot(1,3,2)
    plt.plot(epoch_list, act_loss_list, label='train')
    plt.plot(epoch_list, val_act_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('action')

    #Object Loss
    obj_loss_list = []
    val_obj_loss_list = []
    for loss in loss_data['loss']['object']['train']:
        obj_loss_list.append(loss)
    for val_loss in loss_data['loss']['object']['val']:
        val_obj_loss_list.append(val_loss)
    plt.subplot(1,3,3)
    plt.plot(epoch_list, obj_loss_list, label='train')
    plt.plot(epoch_list, val_obj_loss_list , label='val')
    plt.legend(loc='upper right')
    plt.title('object')

    plt.savefig(os.path.join(model_path, "_".join(list(map(str, model_param))) +'_loss.png'))

    #Accuracy Plot
    plt.figure(1)
    plt.figure(figsize=(10, 5))
    plt.suptitle('Accuracy')

    #Action Accuracy
    act_acc_list = []
    val_act_acc_list = []
    for acc in loss_data['acc']['action']['train']:
        act_acc_list.append(acc)
    for val_acc in loss_data['acc']['action']['val']:
        val_act_acc_list.append(val_acc)
    plt.subplot(1,2,1)
    plt.plot(epoch_list, act_acc_list, label='train')
    plt.plot(epoch_list, val_act_acc_list , label='val')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.title('action')

    #Object Accuracy
    obj_acc_list = []
    val_obj_acc_list = []
    for acc in loss_data['acc']['object']['train']:
        obj_acc_list.append(acc)
    for val_acc in loss_data['acc']['object']['val']:
        val_obj_acc_list.append(val_acc)
    plt.subplot(1,2,2)
    plt.plot(epoch_list, obj_acc_list, label='train')
    plt.plot(epoch_list, val_obj_acc_list , label='val')
    plt.ylim([0, 1])
    plt.legend(loc='upper right')
    plt.title('object')

    plt.savefig(os.path.join(model_path, "_".join(list(map(str, model_param))) +'_acc.png'))
    if show_result:
        plt.show()

    saved_path = os.path.join(model_path, model_name)

    saved_model = ActionModel(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim)
    saved_model.load_state_dict(torch.load(saved_path))
    saved_model.to(device)

    for param in saved_model.parameters():
        param.requires_grad = False

    data_test = ActionDataset(os.path.join(data_dir, 'action','test'))
    data_test_loader = DataLoader(data_test, 1)

    loss_ce_action = nn.CrossEntropyLoss().to(device)
    loss_ce_object = nn.CrossEntropyLoss().to(device)

    act_num_total = [0, 0, 0]
    act_num_acc = [0, 0, 0]

    obj_num_acc = 0

    saved_model.eval()
    for test_data in data_test_loader:
        test_input, test_target, test_info = test_data
        pred_action_prob, pred_object_prob = saved_model(test_input)

        target_action_prob, target_node_scores = test_target['action'].to(device), test_target['object'].to(device)
        
        test_act_label = torch.argmax(target_action_prob, dim=1).to(device)
        test_obj_label = torch.argmax(target_node_scores, dim=1).to(device)

        L_action = loss_ce_action(pred_action_prob, test_act_label)
        L_object = loss_ce_object(pred_object_prob, test_obj_label)
        L_total = L_action + L_object


        print("#########################################")
        print("data info:")
        print("demo type:", test_info['demo'])
        print("order:", test_info['order'])
        print("step:", test_info['step'].item())
        print("--------------------------------")
        print("Loss:")
        print("--------------------------------")
        print("L_total:", L_total.item())
        print("\n")
        print("pred_action_score:", F.softmax(pred_action_prob, dim=-1))
        print("target_action_prob:",target_action_prob)
        print("L_action:", L_action.item())

        act_num_total[test_act_label.item()] += 1
        print("Action Prediction Result:")
        if torch.argmax(pred_action_prob, dim=1) == test_act_label:
            act_num_acc[test_act_label.item()] += 1
            print("Success!")
        else:
            print("Failed TT")

        print("\n")
        print("pred_object_score:", F.softmax(pred_object_prob, dim=-1))
        print("target_object_prob:",target_node_scores)
        print("L_object:", L_object.item())
        
        print("Object Prediction Result:")
        if torch.argmax(pred_object_prob, dim=1) == test_obj_label:
            obj_num_acc += 1
            print("Success!")
        else:
            print("Failed TT")

        if check_each:
            input()
        
    print("------------------------")
    print("Accuracy:")
    print("------------------------")
    print("Pick Result: {}/{} corrected".format(act_num_acc[0], act_num_total[0]))
    print("Pick Acc: {:01.4f}".format(act_num_acc[0]/ act_num_total[0] if act_num_total[0]>0 else 0))
    print("------------------------")
    print("Place Result: {}/{} corrected".format(act_num_acc[1], act_num_total[1]))
    print("Place Acc: {:01.4f}".format(act_num_acc[1]/ act_num_total[1] if act_num_total[1]>0 else 0))
    print("------------------------")
    print("Pour Result: {}/{} corrected".format(act_num_acc[2], act_num_total[2]))
    print("Pour Acc: {:01.4f}".format(act_num_acc[2]/ act_num_total[2] if act_num_total[2]>0 else 0))
    print("------------------------")
    print("------------------------")
    print("Action Result: {}/{} corrected".format(sum(act_num_acc), sum(act_num_total)))
    print("Action Acc: {:01.4f}".format(sum(act_num_acc)/sum(act_num_total)))

    print("------------------------")
    print("------------------------")
    print("Target Object Result: {}/{} corrected".format(obj_num_acc, sum(act_num_total)))
    print("Target Object Acc: {:01.4f}".format(obj_num_acc/sum(act_num_total)))


def inference_sequence_custom(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num=None, check_each = False):
    check_each = True
    model_param = [data_dir, hidden_dim, num_epoch, batch_size, lr]
    model_path = os.path.join(os.getcwd(), "result", "action","_".join(list(map(str, model_param))))

    if infer_num is not None:
        model_name = 'GP_model_{}.pt'.format(infer_num)
    else:
        model_name = 'GP_model_best.pt'

    saved_path = os.path.join(model_path, model_name)

    saved_model = ActionModel(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim)
    saved_model.load_state_dict(torch.load(saved_path))
    saved_model.to(device)

    for param in saved_model.parameters():
        param.requires_grad = False

    data_test = ActionDataset(os.path.join(data_dir, 'action','test'))
    data_test_loader = DataLoader(data_test, 1)
    saved_model.eval()
    goal_edge_attr = None
    for test_data in data_test_loader:
        test_input, test_target, test_info = test_data

    for test_data in data_test_loader:
        test_input, test_target, test_info = test_data
        print(test_input['edge_index'].shape)
        print(test_input['edge_index'])
        print(test_input['edge_attr'].shape)
        print(test_input['edge_attr'])
        input()
        print("#########################################")
        print("data info:")
        print("demo type:", test_info['demo'])
        print("order:", test_info['order'])
        print("step:", test_info['step'].item())
        print("--------------------------------")
        state_edge_index = test_input['edge_index'].to(device)
        #state_edge_attr = test_input['edge_attr'].to(device)

        print("--------------------------------")
        print("input state_edge_attr:\n")
        print(test_input['edge_attr'][:,:7])
        print("--------------------------------")
        print("goal state_edge_attr:\n")
        print(test_input['edge_attr'][:,7:])
        #print(goal_edge_attr)
        #test_input['edge_attr'][:, 7:] = goal_edge_attr
        #edge update
        goal_planned = False
        num_plan = 1
        while goal_planned is False:
            print("plan number", num_plan)
            pred_action_prob, pred_object_prob = saved_model(test_input)
            action_code = int(torch.argmax(pred_action_prob, dim=1).item())
            object_code = int(torch.argmax(pred_object_prob, dim=1).item())

                # 
                # action X object prob => table 만들기
                # 

            act_obj_prob_table = torch.matmul(torch.transpose(pred_action_prob, 0, 1), pred_object_prob)
            print(act_obj_prob_table.shape)
            sorted_table, indices = torch.sort(act_obj_prob_table.view(-1), descending=True)
            
            for idx in range(120):#임의로 제한
                action_code, object_code = divmod(indices[idx].item(), pred_object_prob.size(-1))
                print("Predicted action and object:")
                print(action_code)
                print(object_code)
                updated_graph = graph_dynamics(test_input, action_code, object_code)
                if updated_graph is not None:#feasible한 경우
                    test_input = updated_graph
                    num_plan+=1
                    break
                else:#not feasible -> table상에서 다음으로 높은 (action-obj)를 실행
                    print("Predicted action is not feasible")
                    print("Try next promising action")
                    continue

            goal_edge_attr = test_input['edge_attr'][:,7:]
            state_edge_attr = test_input['edge_attr'][:,:7]
            #업데이트된 state_edge_attr가 goal_edge_attr와 같으면 => 성공
            if torch.equal(state_edge_attr, goal_edge_attr):
                goal_planned = True
                print('plan success')
                break
            #아니면 => 한스텝 더 플래닝
            test_input['edge_attr'][:, :7] = state_edge_attr
            print("plan one more step")
            input()
            if num_plan > 15: #임의로 제한
                break
        if check_each:
            input()      

def graph_dynamics(test_input, action_code, object_code):
    #graph data(x, edge_index, edge_attr), action종류, target object 입력 받아서
    #입력된 action(obj) 실행해서 업데이트된 graph data(x, edge_index, edge_attr) 리턴
    node_features = test_input['x']
    state_edge_index = test_input['edge_index']
    state_edge_attr = test_input['edge_attr'][:, :7]

    robot_hand_code = 0
    #
    # action에 대한 feasibility까지 체크해줘야한다!
    #

    #Pick
    if action_code == 0:
        print("Pick execute")
        print("target object:", object_code)
        #로봇핸드에 물건이들어있는지 체크 -> 있는상태면 수행X
        if find_in_hand(state_edge_index, state_edge_attr) is not None:
            print("Another obj is already grasped in hand\nPick Failed")
            return None
        else: #로봇 핸드가 비어있다
            if int(node_features[object_code, 5].item()) != 1:
                print("Target object is not graspable\nPick Failed")
                return None
            else:#object가 graspable한 물체
                if is_in_bowl(state_edge_index, state_edge_attr, object_code) is not None:
                    #object가 이미 bowl안에 들어있으면 Pick 불가능하도록
                    print("Target object is in bowl\nPick Falied")
                    return None
                else:#not in bowl   
                    #target object와 관련된 on relation제거
                    state_edge_attr = remove_on(state_edge_index, state_edge_attr, object_code)

                    #로봇핸드와 target object 사이에 grasp relation 설정
                    if find_ea_index(state_edge_index, object_code, robot_hand_code) is not None:
                        state_edge_attr[find_ea_index(state_edge_index, object_code, robot_hand_code), 4] = torch.tensor([1], dtype=torch.float32)
                    if find_ea_index(state_edge_index, robot_hand_code, object_code) is not None:
                        state_edge_attr[find_ea_index(state_edge_index, robot_hand_code, object_code), 5] = torch.tensor([1], dtype=torch.float32)
                    #target object의 포지션을 robot hand 포지션으로 업데이트
                    node_features[object_code, 6:] = node_features[robot_hand_code, 6:]
                    #target object가 bowl인 경우 -> 그 안에 들어있는 box들의 포지션도 같이 업데이트
                    if int(node_features[object_code, 0].item()) == 1:#bowl object
                        in_bowl_list = find_in_bowl(state_edge_index, state_edge_attr, object_code)
                        for in_bowl_code in in_bowl_list:
                            node_features[in_bowl_code, 6:] = node_features[robot_hand_code, 6:]
  
    #Place
    elif action_code == 1:
        print("Place execute")
        print("target object:", object_code)

        in_hand_code = find_in_hand(state_edge_index, state_edge_attr)
        if in_hand_code is None:#로봇 핸드에 물체 없으면 실행불가
            print("There is no obj in robot hand\nPlace Failed")
            return None
        else:#로봇 핸드에 물체 집혀 있는 상태
            #in_hand object가 bowl이고, target object가 region이 아닌경우
            #bowl은 region위에만 place할수 있기때문에 실행 불가
            if int(node_features[in_hand_code, 0].item()) == 1 and int(node_features[object_code, 3].item()) != 1:
                print("Bowl object can be placed on Region only\nPlace Failed")
                return None
            else:
                #로봇 핸드와 in_hand obj 사이의 grasp relation을 제거
                if find_ea_index(state_edge_index, in_hand_code, robot_hand_code) is not None:
                    state_edge_attr[find_ea_index(state_edge_index, in_hand_code, robot_hand_code), 4] = torch.tensor([0], dtype=torch.float32)
                if find_ea_index(state_edge_index, robot_hand_code, in_hand_code) is not None:
                    state_edge_attr[find_ea_index(state_edge_index, robot_hand_code, in_hand_code), 5] = torch.tensor([0], dtype=torch.float32)
                
                # in_hand obj가 box이면서 target obj가 bowl인 경우
                # -> in relation 추가
                if int(node_features[in_hand_code, 1].item()) == 1 and int(node_features[object_code, 0].item()) == 1:#in relations
                    print("in hand obj: box, target obj: bowl")
                    if find_ea_index(state_edge_index, in_hand_code, object_code) is not None:
                        state_edge_attr[find_ea_index(state_edge_index, in_hand_code, object_code), 2] = torch.tensor([1], dtype=torch.float32)
                    if find_ea_index(state_edge_index, object_code, in_hand_code) is not None:
                        state_edge_attr[find_ea_index(state_edge_index, object_code, in_hand_code), 3] = torch.tensor([1], dtype=torch.float32)
                # 나머지 경우 -> on relation 추가
                else: #on relations
                    if find_ea_index(state_edge_index, in_hand_code, object_code) is not None:
                        state_edge_attr[find_ea_index(state_edge_index, in_hand_code, object_code), 0] = torch.tensor([1], dtype=torch.float32)
                    if find_ea_index(state_edge_index, object_code, in_hand_code) is not None:
                        state_edge_attr[find_ea_index(state_edge_index, object_code, in_hand_code), 1] = torch.tensor([1], dtype=torch.float32)
                # in_hand obj의 포지션을 target obj 포지션으로 업데이트
                node_features[in_hand_code, 6:] = node_features[object_code, 6:]
                # in_hand obj가 bowl인 경우:
                # 안에 들어있는 box들의 포지션도 업데이트
                if int(node_features[in_hand_code, 0].item()) == 1:#bowl object
                    in_bowl_list = find_in_bowl(state_edge_index, state_edge_attr, in_hand_code)
                    for in_bowl_code in in_bowl_list:
                        node_features[in_bowl_code, 6:] = node_features[object_code, 6:]
    #Pour
    elif action_code == 2:
        print("Pour execute")
        print("target object:", object_code)

        in_hand_code = find_in_hand(state_edge_index, state_edge_attr)
        
        # 로봇 핸드에 물체 없는경우 -> pour 불가능
        if in_hand_code is None:
            print("There is no obj in robot hand\nPour Failed")
            return None
        # in_hand obj가 bowl이 아닌 경우 -> pour 불가능
        elif int(node_features[in_hand_code, 0].item()) != 1:#not bowl
            print("The obj in robot hand is not bowl\nPour Failed")
            return None
        # 로봇 핸드에 bowl이 집혀있는 경우
        else:#in_hand_object = bowl
            print("The obj in hand is:", in_hand_code)
            #target obj가 bowl이 아닌 경우 -> pour 불가능
            if int(node_features[object_code, 0].item()) != 1:
                print("The target object is not bowl\nPour Failed")
                return None
            #target obj도 bowld인 경우
            else:#object_code = target bowl
                #in_hand bowl 안에 있는 box 체크
                in_bowl_list = find_in_bowl(state_edge_index, state_edge_attr, in_hand_code)
                print("Objects in grasped bowl:", in_bowl_list)
                # #bowl이 비어있는 경우 -> pour 불가능?
                # if len(in_bowl_list) == 0:
                #     print("Grasped bowl is empty\nPour Failed")
                #     return None
                # else:
                # in_hand bowl 안에 있는 box들에 대해서 각각 수행
                for in_bowl_code in in_bowl_list:
                    # in_bowl box와 in_hand bowl사이에 in relation 제거
                    if find_ea_index(state_edge_index, in_bowl_code, in_hand_code) is not None:
                        state_edge_attr[find_ea_index(state_edge_index, in_bowl_code, in_hand_code), 2] = torch.tensor([0], dtype=torch.float32)
                    if find_ea_index(state_edge_index, in_hand_code, in_bowl_code) is not None:
                        state_edge_attr[find_ea_index(state_edge_index, in_hand_code, in_bowl_code), 3] = torch.tensor([0], dtype=torch.float32)
                    # in_bowl box와 target bowl사이에 in relation 추가
                    if find_ea_index(state_edge_index, in_bowl_code, object_code) is not None:
                        state_edge_attr[find_ea_index(state_edge_index, in_bowl_code, object_code), 2] = torch.tensor([1], dtype=torch.float32)
                    if find_ea_index(state_edge_index, object_code, in_bowl_code) is not None:
                        state_edge_attr[find_ea_index(state_edge_index, object_code, in_bowl_code), 3] = torch.tensor([1], dtype=torch.float32)
                    # in_bowl box의 포지션을 target bowl 포지션으로 업데이트
                    node_features[in_bowl_code, 6:] = node_features[object_code, 6:]

    else:
        print("Predicted action code is wrong\nAction Failed")
        return None

    test_input['x'] = node_features
    test_input['edge_index'] = state_edge_index
    test_input['edge_attr'][:,:7] = state_edge_attr
    return test_input

def find_ea_index(state_edge_index, src, dest):
    # 주어진 엣지 정보(src, dest) 에 해당하는 edge index값을 리턴
    for idx in range(state_edge_index.size(-1)):
        pair = list(map(int, state_edge_index[:, idx].tolist()))
        if pair == [src, dest]:
            return idx
    return None

def remove_on(state_edge_index, state_edge_attr, obj):
    # 주어진 obj에 대해 on relation을 모두 제거
    max_obj_num = 33
    for i in range(max_obj_num):
        ea_idx = find_ea_index(state_edge_index, obj, i)
        state_edge_attr[ea_idx, 0] = 0
        ea_idx = find_ea_index(state_edge_index, i, obj)
        state_edge_attr[ea_idx, 1] = 0
    return state_edge_attr

def find_in_hand(state_edge_index, state_edge_attr):
    #현재 로봇 핸드에 집혀있는 물체 리턴
    max_obj_num = 33
    hand = 0
    for i in range(max_obj_num):
        ea_idx = find_ea_index(state_edge_index, hand, i)
        if ea_idx is not None:
            if int(state_edge_attr[ea_idx, 5].item()) == 1:
                return i
            
        ea_idx = find_ea_index(state_edge_index, i, hand)
        if ea_idx is not None:
            if int(state_edge_attr[ea_idx, 4].item()) == 1:
                return i
    return None

def find_in_bowl(state_edge_index, state_edge_attr, bowl_code):
    #주어진 bowl_code에 해당하는 bowl 안에 들어있는 box들의 리스트를 리턴
    max_obj_num = 33
    in_bowl_code_set = set()
    for i in range(max_obj_num):
        ea_idx = find_ea_index(state_edge_index, bowl_code, i)
        if ea_idx is not None:
            if int(state_edge_attr[ea_idx, 3].item()) == 1:
                in_bowl_code_set.add(i)
                continue
            
        ea_idx = find_ea_index(state_edge_index, i, bowl_code)
        if ea_idx is not None:
            if int(state_edge_attr[ea_idx, 2].item()) == 1:
                in_bowl_code_set.add(i)
                continue
    return list(in_bowl_code_set)

def is_in_bowl(state_edge_index, state_edge_attr, target_object):
    #주어진 target_object가 bowl 안에 들어있는 상태인지 리턴
    #들어있는 상태면 해당 bowl까지 출력
    print("checking target object is in bowl..")
    max_obj_num = 33
    for i in range(max_obj_num):
        ea_idx = find_ea_index(state_edge_index, target_object, i)
        if ea_idx is not None:
            if int(state_edge_attr[ea_idx, 2].item()) == 1:
                print("in ",i)
                return i
            
        ea_idx = find_ea_index(state_edge_index, i, target_object)
        if ea_idx is not None:
            if int(state_edge_attr[ea_idx, 3].item()) == 1:
                return i
    return None