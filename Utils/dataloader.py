import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F 
import numpy as np
import pickle
import torch
from Utils.evaluation import *

## helper functions
def load_pickle(path, filename):
    with open(path + filename, 'rb') as f:
        obj = pickle.load(f)

    return obj

def to_np(x):
    return x.data.cpu().numpy()


def dict_set(base_dict, u_id, i_id, val):
    if u_id in base_dict:
        base_dict[u_id][i_id] = val
    else:
        base_dict[u_id] = {i_id: val}


def is_visited(base_dict, u_id, i_id):
    if u_id in base_dict and i_id in base_dict[u_id]:
        return True
    else:
        return False


def list_to_dict(base_list):
    result = {}
    for u_id, i_id, value in base_list:
        dict_set(result, u_id, i_id, value)
    
    return result


def dict_to_list(base_dict):
    result = []

    for u_id in base_dict:
        for i_id in base_dict[u_id]:
            result.append((u_id, i_id, 1))
    
    return result

def read_file(file_path):
   """
   读取 Amazon CSV 文件 (优化版本)
   """
   import pandas as pd
   
   # 读取CSV文件
   df = pd.read_csv(file_path)
   
   # 打印原始数据的前10行
   print("原始数据的前10行:")
   print(df[['reviewerID', 'asin', 'overall']].head(10))
   print("\n")
   
   # 创建映射字典
   user_map = {user: idx for idx, user in enumerate(df['reviewerID'].unique())}
   item_map = {item: idx for idx, item in enumerate(df['asin'].unique())}
   
   # 获取用户数和物品数
   user_count = len(user_map)
   item_count = len(item_map)
   
   print(f"User count: {user_count}, Item count: {item_count}")
   
   # 向量化操作：直接将所有ID映射为整数
   user_ids = df['reviewerID'].map(user_map)
   item_ids = df['asin'].map(item_map)
   ratings = df['overall']
   
   # 直接创建交互列表
   total_interactions = list(zip(user_ids, item_ids, ratings))
   
   # 打印映射后的前10个交互
   print("\n映射后的前10个交互:")
   print(total_interactions[:10])
   
   # 创建反向映射（用于展示对应关系）
   reverse_user_map = {v: k for k, v in user_map.items()}
   reverse_item_map = {v: k for k, v in item_map.items()}
   
   # 打印映射对应关系
   print("\n映射对应关系示例（前10个交互）:")
   for i, (u_id, i_id, rating) in enumerate(total_interactions[:10]):
       original_user = reverse_user_map[u_id]
       original_item = reverse_item_map[i_id]
       print(f"交互{i+1}: {original_user}({u_id}) - {original_item}({i_id}) - 评分{rating}")
   
   return user_count, item_count, total_interactions

def load_data(file_path='reviews_Digital_Music_5.csv', test_ratio=0.2, random_seed=0):
    """
    加载数据并分割训练集、验证集和测试集
    """
    np.random.seed(random_seed)

    # 读取CSV文件
    u_count, i_count, total_int_tmp = read_file(file_path)

    # 获取用户和物品的交互次数统计
    u_count_dict, i_count_dict = get_count_dict(total_int_tmp)
    
    # 过滤交互数据（count_filtering=[0, 0],用户至少0次交互，物品至少0次交互）
    u_count, i_count, total_ints = get_total_ints(
        total_int_tmp, 
        u_count_dict, 
        i_count_dict, 
        is_implicit=True,  # 隐式反馈
        count_filtering=[0, 0]  # 用户,物品不过滤
    )
    
    # 转换为字典格式
    total_mat = list_to_dict(total_ints)

    # 分割训练集、验证集和测试集
    train_mat, valid_mat, test_mat = {}, {}, {}

    for user in total_mat:
        items = list(total_mat[user].keys())
        np.random.shuffle(items)
        # train:valid:test = 6:2:2 for each user
        num_test_items = int(len(items) * test_ratio)
        test_items = items[:num_test_items]
        valid_items = items[num_test_items: num_test_items*2]
        train_items = items[num_test_items*2:]

        for item in test_items:
            dict_set(test_mat, user, item, total_mat[user][item])  # 保留原始评分

        for item in valid_items:
            dict_set(valid_mat, user, item, total_mat[user][item])

        for item in train_items:
            dict_set(train_mat, user, item, total_mat[user][item])
    
    
    # 打印数据集统计信息
    print("\n数据集统计信息:")
    print(f"训练集 - 用户数: {len(train_mat):,}, 交互数: {sum(len(items) for items in train_mat.values()):,}")
    print(f"验证集 - 用户数: {len(valid_mat):,}, 交互数: {sum(len(items) for items in valid_mat.values()):,}")
    print(f"测试集 - 用户数: {len(test_mat):,}, 交互数: {sum(len(items) for items in test_mat.values()):,}")

    # 创建物品-用户倒排表
    train_mat_R = {}
    for u in train_mat:
        for i in train_mat[u]:
            dict_set(train_mat_R, i, u, train_mat[u][i])
    
    # 过滤验证集和测试集中的冷启动物品
    # for u in list(valid_mat.keys()):
    #     for i in list(valid_mat[u].keys()):
    #         if i not in train_mat_R:
    #             del valid_mat[u][i]
    #     if len(valid_mat[u]) == 0:
    #         del valid_mat[u]
    #         del test_mat[u]
            
    # for u in list(test_mat.keys()):
    #     for i in list(test_mat[u].keys()):
    #         if i not in train_mat_R:
    #             del test_mat[u][i]
    #     if len(test_mat[u]) == 0:
    #         del test_mat[u]
    #         del valid_mat[u]
    
    # 转换训练集为列表格式
    train_ints = []
    for u in train_mat:
        for i in train_mat[u]:
            train_ints.append([u, i, train_mat[u][i]])
            
    return u_count, i_count, train_mat, train_ints, valid_mat, test_mat

def get_count_dict(total_ints, spliter="\t"):

    u_count_dict, i_count_dict = {}, {}

    for line in total_ints:
        u, i, rating = line
        u, i, rating = int(u), int(i), float(rating)

        if u in u_count_dict:
            u_count_dict[u] += 1
        else: 
            u_count_dict[u] = 1

        if i in i_count_dict:
            i_count_dict[i] += 1
        else: 
            i_count_dict[i] = 1

    return u_count_dict, i_count_dict


def get_total_ints(total_int_tmp, u_count_dict, i_count_dict, is_implicit=True, count_filtering = [10, 10], spliter="\t"):

    total_ints = []
    u_dict, i_dict = {}, {}
    u_count, i_count = 0, 0

    for line in total_int_tmp:
        u, i, rating = line
        u, i, rating = int(u), int(i), float(rating)

        # count filtering
        if u_count_dict[u] < count_filtering[0]:
            continue
        if i_count_dict[i] < count_filtering[1]:
            continue

        # u indexing
        if u in u_dict:
            u_id = u_dict[u]
        else:
            u_id = u_count
            u_dict[u] = u_id
            u_count += 1

        # i indexing
        if i in i_dict:
            i_id = i_dict[i]
        else:
            i_id = i_count
            i_dict[i] = i_id
            i_count += 1

        if is_implicit:
            rating = 1.

        total_ints.append((u_id, i_id, rating))

    return u_count + 1, i_count + 1, total_ints




class train_dataset(data.Dataset):
    def __init__(self, user_count, item_count, rating_mat, num_ns, interactions, exception_interactions=[]):
        super(train_dataset, self).__init__()
        
        self.user_count = user_count
        self.item_count = item_count
        self.rating_mat = rating_mat
        self.num_ns = num_ns
        self.interactions = interactions
        self.exception_interactions = exception_interactions

        self.R = torch.zeros((user_count, item_count))
        for user in rating_mat:
            items = list(rating_mat[user].keys())
            self.R[user][items] = 1.

        if len(exception_interactions) > 0:
            self.exception_mat = {}
            for u, i, _ in exception_interactions:
                dict_set(self.exception_mat, u, i, 1)
        
    def negative_sampling(self):
        
        self.train_arr = []
        sample_list = np.random.choice(list(range(self.item_count)), size = 10 * len(self.interactions) * self.num_ns)
        
        sample_idx = 0
        for user, pos_item, _ in self.interactions:
            ns_count = 0
            
            while True:
                neg_item = sample_list[sample_idx]
                if len(self.exception_interactions) > 0:
                    if not is_visited(self.rating_mat, user, neg_item) and not is_visited(self.exception_mat, user, neg_item) :
                        self.train_arr.append((user, pos_item, neg_item))
                        sample_idx += 1
                        ns_count += 1
                        if ns_count == self.num_ns:
                            break
                else:
                    if not is_visited(self.rating_mat, user, neg_item):
                        self.train_arr.append((user, pos_item, neg_item))
                        sample_idx += 1
                        ns_count += 1
                        if ns_count == self.num_ns:
                            break					
                        
                sample_idx += 1
    
    def __len__(self):
        return len(self.interactions) * self.num_ns
        
    def __getitem__(self, idx):

        return {'u': self.train_arr[idx][0], 
                'p': self.train_arr[idx][1], 
                'n': self.train_arr[idx][2]}

    def get_user_side_mask(self, batch_user):
        return torch.index_select(self.R, 0 , batch_user.cpu())



class test_dataset(data.Dataset):
    def __init__(self, user_count, item_count, valid_mat, test_mat, batch_size=64):
        super(test_dataset, self).__init__()

        self.user_count = user_count
        self.item_count = item_count
        self.user_list = torch.LongTensor([i for i in range(user_count)])

        self.valid_mat = valid_mat
        self.test_mat = test_mat
        self.batch_size = batch_size

        self.batch_start = 0

    def get_next_batch_users(self):
        batch_start = self.batch_start
        batch_end = self.batch_start + self.batch_size

        if batch_end >= self.user_count:
            batch_end = self.user_count
            self.batch_start = 0
            return self.user_list[batch_start: batch_end], True
        else:
            self.batch_start += self.batch_size
            return self.user_list[batch_start: batch_end], False




