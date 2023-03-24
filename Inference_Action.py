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
        if test_info['step'].item() == 7:
            goal_edge_attr = test_input['edge_attr'][:, 7:]
            break

    for test_data in data_test_loader:
        test_input, test_target, test_info = test_data
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
        #print(test_input['edge_attr'][:,7:])
        print(goal_edge_attr)
        test_input['edge_attr'][:, 7:] = goal_edge_attr
        #edge update
        goal_planned = False
        num_plan = 1
        while goal_planned is False:
            print("plan number", num_plan)
            pred_action_prob, pred_object_prob = saved_model(test_input)
            action_code = int(torch.argmax(pred_action_prob, dim=1).item())
            object_code = int(torch.argmax(pred_object_prob, dim=1).item())

            #goal_edge_attr = test_input['edge_attr'][:,7:]
            state_edge_attr = test_input['edge_attr'][:,:7]
            print("--------------------------------")
            print("updated state_edge_attr:\n")
            state_edge_attr = graph_dynamics(state_edge_index, state_edge_attr, action_code, object_code)
            print(state_edge_attr)
            num_plan += 1

            if torch.equal(state_edge_attr, goal_edge_attr):
                goal_planned = True
                print('plan success')
                break
            test_input['edge_attr'][:, :7] = state_edge_attr
            print("plan one more step")
            input()
            if num_plan > 15:
                break
        if check_each:
            input()

def inference_sequence(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num=None, check_each = False):
    
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
    for test_data in data_test_loader:
        test_input, test_target, test_info = test_data
        print("#########################################")
        print("data info:")
        print("demo type:", test_info['demo'])
        print("order:", test_info['order'])
        print("step:", test_info['step'].item())
        print("--------------------------------")
        state_edge_index = test_input['edge_index'].to(device)
        #state_edge_attr = test_input['edge_attr'].to(device)

        #print("--------------------------------")
        #print("input state_edge_attr:\n")
        #print(test_input['edge_attr'][:,:7])
        #print("--------------------------------")
        #print("goal state_edge_attr:\n")
        #print(test_input['edge_attr'][:,7:])
        #edge update
        goal_planned = False
        num_plan = 1
        plan_list = []
        res = recursive_plan(test_input, saved_model, plan_list, goal_reached=False, plan_step = 0)
        '''
        while goal_planned is False:
            print("plan number", num_plan)
            pred_action_prob, pred_object_prob = saved_model(test_input)
            act_obj_prob_table = torch.matmul(torch.transpose(pred_action_prob, 0, 1), pred_object_prob)
            sorted_table, indices = torch.sort(act_obj_prob_table.view(-1), descending=True)
            for idx  in range((pred_action_prob.size(-1))*(pred_object_prob.size(-1))):
                action_code, object_code = divmod(indices[idx].item(), pred_object_prob.size(-1))
            
            #_action_code = torch.argsort(pred_action_prob, dim=1)
            #_object_code = torch.argsort(pred_object_prob, dim=1)
            #action_code = int(torch.argmax(pred_action_prob, dim=1).item())
            #object_code = int(torch.argmax(pred_object_prob, dim=1).item())

            goal_edge_attr = test_input['edge_attr'][:,7:]
            state_edge_attr = test_input['edge_attr'][:,:7]
            #print("--------------------------------")
            #print("updated state_edge_attr:\n")
            state_edge_attr = graph_dynamics(state_edge_index, state_edge_attr, action_code, object_code)
            #print(state_edge_attr)
            num_plan += 1

            if torch.equal(state_edge_attr, goal_edge_attr):
                goal_planned = True
                print('plan success')
                break
            test_input['edge_attr'][:, :7] = state_edge_attr
            print("plan one more step")
            input()
            if num_plan > 15:
                break
        '''
        if check_each:
            input()
        
def graph_plan(test_input, saved_model, goal_reached, plan_step):
    if torch.equal(test_input['edge_attr'][:, :7], test_input['edge_attr'][:, 7:]):
        goal_reached = True
        return goal_reached
    else:
        plan_step+=1
        pred_action_prob, pred_object_prob = saved_model(test_input)
        act_obj_prob_table = torch.matmul(torch.transpose(pred_action_prob, 0, 1), pred_object_prob)
        sorted_table, indices = torch.sort(act_obj_prob_table.view(-1), descending=True)
        for idx  in range((pred_action_prob.size(-1))*(pred_object_prob.size(-1))):
            action_code, object_code = divmod(indices[idx].item(), pred_object_prob.size(-1))
            updated_state = graph_dynamics(test_input['edge_index'], test_input['edge_attr'][:, :7], action_code, object_code)
            if updated_state is not None:#if updated state is feasible

                test_input['edge_attr'][:, :7] = updated_state
                goal_reached = graph_plan(test_input, saved_model, goal_reached, plan_step)

def recursive_plan(test_input, saved_model, plan_list, goal_reached, plan_step):
    print("recursive")
    plan_step += 1
    print("plan_step", len(plan_list))
    print(plan_list)
    if len(plan_list) > 10:
        plan_step -= 1
        return plan_list.pop()
        
    input()
    idx = 0
    pred_action_prob, pred_object_prob = saved_model(test_input)
    act_obj_prob_table = torch.matmul(torch.transpose(pred_action_prob, 0, 1), pred_object_prob)
    sorted_table, indices = torch.sort(act_obj_prob_table.view(-1), descending=True)
    idx=0
    while idx < (pred_action_prob.size(-1))*(pred_object_prob.size(-1)):
        action_code, object_code = divmod(indices[idx].item(), pred_object_prob.size(-1))
        updated_state = graph_dynamics(test_input['edge_index'], test_input['edge_attr'][:, :7], action_code, object_code)
        if updated_state is None:
            idx+=1
            continue
        else:
            plan_list.append((action_code, object_code))
            if torch.equal(updated_state, test_input['edge_attr'][:, 7:]):
                return plan_list
            else:
                test_input['edge_attr'][:,:7] = updated_state
                return recursive_plan(test_input, saved_model,plan_list, goal_reached, plan_step)


def graph_dynamics(state_edge_index, state_edge_attr, action_code, object_code):
    zero_ea = torch.zeros(7)#edge_attr size
    #edge_attr sturcture
    #[rel_on_right,rel_on_left,rel_in_right,rel_in_left,rel_in_grasp,rel_grasp,rel_attach]
    #print(action_code)
    #print(object_code)

    robot_hand_code = 0
    table_code = 8
    #in_hand_code = find_in_hand(state_edge_index, state_edge_attr)
    #print(in_hand_code)
    #print(state_edge_index)

    #print(find_ea_index(state_edge_index, object_code, robot_hand_code))
    #print(find_ea_index(state_edge_index, in_hand_code, robot_hand_code))
    #print(find_ea_index(state_edge_index, object_code, table_code))
    #Pick
    if action_code == 0:
        print("Pick execute")
        print("target object:", object_code)
        if find_in_hand(state_edge_index, state_edge_attr) is not None:
            print("Another obj is already grasped in hand\nPick Failed")
            return None
        else:
            state_edge_attr = remove_on(state_edge_index, state_edge_attr, object_code)

            if find_ea_index(state_edge_index, object_code, robot_hand_code) is not None:
                state_edge_attr[find_ea_index(state_edge_index, object_code, robot_hand_code), 4] = torch.tensor([1], dtype=torch.float32)
            if find_ea_index(state_edge_index, robot_hand_code, object_code) is not None:
                state_edge_attr[find_ea_index(state_edge_index, robot_hand_code, object_code), 5] = torch.tensor([1], dtype=torch.float32)
    #Place
    elif action_code == 1:
        print("Place execute")
        print("target object:", object_code)

        in_hand_code = find_in_hand(state_edge_index, state_edge_attr)
        if in_hand_code is None:
            print("There is no obj in robot hand\nPlace Failed")
            return None
        else:
            if find_ea_index(state_edge_index, in_hand_code, robot_hand_code) is not None:
                state_edge_attr[find_ea_index(state_edge_index, in_hand_code, robot_hand_code), 4] = torch.tensor([0], dtype=torch.float32)
            if find_ea_index(state_edge_index, robot_hand_code, in_hand_code) is not None:
                state_edge_attr[find_ea_index(state_edge_index, robot_hand_code, in_hand_code), 5] = torch.tensor([0], dtype=torch.float32)
            if find_ea_index(state_edge_index, in_hand_code, object_code) is not None:
                state_edge_attr[find_ea_index(state_edge_index, in_hand_code, object_code), 0] = torch.tensor([1], dtype=torch.float32)
            if find_ea_index(state_edge_index, object_code, in_hand_code) is not None:
                state_edge_attr[find_ea_index(state_edge_index, object_code, in_hand_code), 1] = torch.tensor([1], dtype=torch.float32)
    #Pour
    elif action_code == 2:
        print("Pour execute")
        pass
    #Mix
    elif action_code == 3:
        print("Mix execute")
        pass

    return state_edge_attr

def find_ea_index(state_edge_index, src, dest):
    for idx in range(state_edge_index.size(-1)):
        pair = list(map(int, state_edge_index[:, idx].tolist()))
        if pair == [src, dest]:
            return idx
    return None

def remove_on(state_edge_index, state_edge_attr, obj):
    max_obj_num = 13
    for i in range(max_obj_num):
        ea_idx = find_ea_index(state_edge_index, obj, i)
        state_edge_attr[ea_idx, 0] = 0
        ea_idx = find_ea_index(state_edge_index, i, obj)
        state_edge_attr[ea_idx, 1] = 0
    return state_edge_attr

def find_in_hand(state_edge_index, state_edge_attr):
    max_obj_num = 13
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