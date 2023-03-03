from Train import *
from Inference import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_action = 3 # [pick, place, pour]
node_feature_size = 6 #노드 feature 크기
edge_feature_size = 26 # 

hidden_dim = 128
num_epoch = 1000
batch_size = 32
lr = 1e-06
data_dir = "stack_mix_fc"
show_result = True
infer_num = None
check_each = False


#train_act_only(device, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size, lr, num_epoch, data_dir)

inference_act_only(device, hidden_dim, num_action, node_feature_size, edge_feature_size, batch_size, lr, num_epoch, data_dir, show_result, infer_num, check_each)