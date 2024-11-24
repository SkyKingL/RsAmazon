import random
import sys
import argparse
import numpy as np

from Utils.evaluation import *
from Utils.dataloader import load_data, train_dataset, test_dataset

import torch
import torch.optim as optim
import torch.utils.data as data

from Models.RSModel import MF

from tqdm import tqdm

import os

# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--dim', type=int, default=6)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_ns', type=int, default=1)

parser.add_argument('--test_ratio', type=float, default=0.20)
parser.add_argument('--random_seed', type=int, default=0)
parser.add_argument('--alpha', type=float, default=1.05)
parser.add_argument('--p', type=int, default=10)
parser.add_argument('--cheekpoint_path', type=str, default="checkpoint/best.pth")

opt = parser.parse_args()

gpu = torch.device('cuda:0') 

random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

alpha = opt.alpha
p = opt.p
K = 100

if not os.path.exists("checkpoint"):
    os.makedirs("checkpoint")

def get_NDCG_u(sorted_list, teacher_t_items, user, k=10):

    with torch.no_grad():
        top_scores = np.asarray([np.exp(-t/10) for t in range(k)])
        top_scores = ((2 ** top_scores)-1)
        
        t_items = teacher_t_items[:k]

        sorted_list_tmp = []
        for item in sorted_list:
            if user in train_mat and item not in train_mat[user]:
                sorted_list_tmp.append(item)
            if len(sorted_list_tmp) == k: break  

        if user not in train_mat:
            sorted_list_tmp = sorted_list

        denom = np.log2(np.arange(2, k + 2))
        dcg_10 = np.sum((np.in1d(sorted_list_tmp[:k], list(t_items)) * top_scores) / denom)
        idcg_10 = np.sum((top_scores / denom)[:k])

        return round(dcg_10 / idcg_10, 4)

#############################################################################################################################
# data load
print("Data loading...")
# 文件路径打印
print(f"文件路径: reviews_Apps_for_Android_5.csv")
user_count, item_count, train_mat, train_interactions, valid_mat, test_mat = load_data(file_path='reviews_Apps_for_Android_5.csv', test_ratio=0.2)

train_dataset = train_dataset(user_count, item_count, train_mat, 1, train_interactions)
test_dataset = test_dataset(user_count, item_count, valid_mat, test_mat)
train_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)

##############################################################################################################################
print("Training...")
# MF model 
model = MF(user_count, item_count, opt.dim, gpu)
model = model.to(gpu)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.reg)

train_losses = []
b_recall = -999
b_result, f_result = -1, -1

es = 0
verbose = 50
last_dist = None
is_first = True
v_results = np.asarray([0, 0, 0, 0, 0, 0])

# 保存模型
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

# 加载模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")

for epoch in range(1000):

    tic1 = time.time()
    train_loader.dataset.negative_sampling()
    B_loss = []

    for mini_batch in tqdm(train_loader, desc=f'Epoch {epoch+1} / 1000'):

        b_u = mini_batch['u'].unique()
        
        mini_batch = {key: value.to(gpu) for key, value in mini_batch.items()}

        model.train()
        output = model(mini_batch)

        b_loss = model.get_loss(output)
  
        B_loss.append(b_loss)
        optimizer.zero_grad()
        b_loss.backward()
        optimizer.step()

    B_loss = torch.mean(torch.stack(B_loss)).data.cpu().numpy()
    train_losses.append(B_loss)

    toc1 = time.time()
    if (epoch+1) % verbose == 0:
        imp = False
        print("="* 50)
        model.eval()
        with torch.no_grad():
            tic2 = time.time()
            e_results, sorted_mat = evaluate(model, gpu, train_loader, test_dataset, return_sorted_mat=True)
            toc2 = time.time()

            if e_results['valid']['R10'] > b_recall: 
                imp = True
                print("Improved! Saving model...")
                save_model(model, opt.cheekpoint_path)
                
                b_recall = e_results['valid']['R10']
                b_result = e_results['valid']
                f_result = e_results['test']
                es = 0						
            else:
                imp = False
                es += 1

            print_result(epoch+1, 1000, B_loss, e_results, is_improved=imp, train_time=toc1-tic1, test_time=toc2-tic2)
        
        print("="* 50)

    if es >= 4: # Early stopping
        print("Early stopping...")
        break
    
# 测试
print("Testing...")
load_model(model, opt.cheekpoint_path)
model.eval()
with torch.no_grad():
    e_results, sorted_mat = evaluate(model, gpu, train_loader, test_dataset, return_sorted_mat=True)
    print_final_result(e_results)