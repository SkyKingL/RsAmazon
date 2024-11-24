import torch
import torch.nn as nn
import torch.nn.functional as F

class MF(nn.Module):
	def __init__(self, user_count, item_count, dim, gpu):
		super(MF, self).__init__()
		self.user_count = user_count
		self.item_count = item_count

		self.user_list = torch.LongTensor([i for i in range(user_count)]).to(gpu)
		self.item_list = torch.LongTensor([i for i in range(item_count)]).to(gpu)

		self.user_emb = nn.Embedding(self.user_count, dim)
		self.item_emb = nn.Embedding(self.item_count, dim)

		nn.init.normal_(self.user_emb.weight, mean=0., std= 0.01)
		nn.init.normal_(self.item_emb.weight, mean=0., std= 0.01)


	def forward(self, mini_batch):

		user = mini_batch['u']
		pos_item = mini_batch['p']
		neg_item = mini_batch['n']

		u = self.user_emb(user)
		i = self.item_emb(pos_item)
		j = self.item_emb(neg_item)

		return u, i, j
	
	'''
	最大化用户对正样本物品的预测得分与负样本物品预测得分之间的差距
 	'''
	def get_loss(self, output):

		h_u, h_i, h_j = output[0], output[1], output[2]
		# 计算正样本得分
		bpr_pos_score = (h_u * h_i).sum(dim=1, keepdim=True)
  		# 计算负样本得分
		bpr_neg_score = (h_u * h_j).sum(dim=1, keepdim=True)
		# BPR loss的核心思想就是：最大化"用户对正样本物品的偏好得分"减去"用户对负样本物品的偏好得分"的差值，
  		# 通过对这个差值进行sigmoid函数映射后取对数再求和，得到一个需要最小化的损失值。
		bpr_loss = -(bpr_pos_score - bpr_neg_score).sigmoid().log().sum()

		return bpr_loss

	def get_embedding(self):

		user = self.user_emb(self.user_list)
		item = self.item_emb(self.item_list)

		return user, item

	def forward_full_items(self, batch_user):
		user = self.user_emb(batch_user)
		item = self.item_emb(self.item_list) 

		return torch.matmul(user, item.T)