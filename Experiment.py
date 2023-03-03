from Train import *
from Inference import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_action = 3 # [pick, place, pour]
node_feature_size = 6 #노드 feature 크기
edge_feature_size = 26 # 
data_dir = "stack_mix_fc"
show_result = False
infer_num = None
check_each = False

num_epoch = 3* 0
hidden_dim_list = [128]
batch_size_list = [32]
#lr_list = [1e-06, 5e-06, 1e-05, 5e-05]
lr_list = [1e-03]

for h in hidden_dim_list:
    for l in lr_list:
        for b in batch_size_list:
            print("########## hidden_dim={}, batch={}, lr={} ##########".format(h, b, l))
            train_act_only(device, h, num_action, node_feature_size, edge_feature_size, b, l, num_epoch, data_dir)
            inference_act_only(device, h, num_action, node_feature_size, edge_feature_size, b, l, num_epoch, data_dir, show_result, infer_num, check_each)