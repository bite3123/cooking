from Train_Action import *
from Inference_Action import *
from Train_Dynamics import *
from Inference_Dynamics import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_action = 4 # [pick, place, pour, mix]
node_feature_size = 7 #
edge_feature_size = 7 # 

hidden_dim = 128
num_epoch = 1001
batch_size = 16
lr = 1e-03
data_dir = "stacking_v4X_posX"
show_result = False
infer_num = None
check_each = True
global_dim = 16
#action / object lr 다르게
#train_action(device, hidden_dim, num_action, node_feature_size, edge_feature_size*2, global_dim, batch_size, lr, num_epoch, data_dir)
#inference_action(device, hidden_dim, num_action, node_feature_size, edge_feature_size*2, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num, check_each)
inference_sequence(device, hidden_dim, num_action, node_feature_size, edge_feature_size*2, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num, check_each)
#dynamics
#train_dynamics(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir)
#inference_dynamics(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num, check_each)
#train_dynamics_test(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir)
#inference_dynamics_test(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num, check_each)