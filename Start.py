from Train_Action import *
from Inference_Action import *
from Train_Dynamics import *
from Inference_Dynamics import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#필요한 hyper parameter 설정
num_action = 3 # [pick, place, pour]
node_feature_size = 12 #nf6+pose6
edge_feature_size = 7 # 

global_dim = 16
hidden_dim = 512
num_epoch = 300
batch_size = 256
lr = 1e-05
data_dir = "stack_mix_clean_pose5" #CollectGraph에서 만든 데이터셋이름과 동일하게 입력
show_result = True #inference 시작할때 결과 plot할지 말지
infer_num = None #inference할 epoch 값을 지정(None이면 best값으로)
check_each = False #inference에서 각 test data마다 stop해서 확인 할 지 여부

#train_action(device, hidden_dim, num_action, node_feature_size, edge_feature_size*2, global_dim, batch_size, lr, num_epoch, data_dir)
#inference_action(device, hidden_dim, num_action, node_feature_size, edge_feature_size*2, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num, check_each)

#dynamics따라서 sequential planning 실험용
inference_sequence_custom(device, hidden_dim, num_action, node_feature_size, edge_feature_size*2, global_dim, batch_size, lr, num_epoch, data_dir, show_result, infer_num, check_each)