from Train import *
from Inference import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_action = 3 # [pick, place, pour]
node_feature_size = 6 #
edge_feature_size = 26 # 

hidden_dim = 128
num_epoch = 7
batch_size = 16
lr = 1e-05
data_dir = "stack_mix_fc_test"
show_result = False
infer_num = None
check_each = False
global_dim = 16


#train_act_only(device, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size, lr, num_epoch, data_dir)
#inference_act_only(device, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size, lr, num_epoch, data_dir, show_result, infer_num, check_each)

#train(device, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size, lr, num_epoch, data_dir)
#inference(device, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size, lr, num_epoch, data_dir, show_result, infer_num, check_each)

train_test(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir)
inference_test(device, hidden_dim, num_action, node_feature_size, edge_feature_size, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num, check_each)