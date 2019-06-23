from __future__ import division
import numpy as np 

class WeakClassifier:

	'''
	for every feature,calculate the all the possible decision_threshold\
	remember 'gt':great than or 'lt':less than. Finally,get a dictionary\
	as dic={'gt':{'0':...},{'1':...},...,'lt':{'0':...},{'1':...},...}\
	the symbol '...' above is a two dimension np.array([[],[],[],...])\
	'''
	def cal_dic(self,X):
		ret_gt={}
		for i in range(X.shape[1]): # 列
			ret_gt[i]=[]  
			for j in range(X.shape[0]): # 行
				temp_threshold=X[j,i]
				temp_line=[]
				for k in range(X.shape[0]):
					if X[k,i]>=temp_threshold:
						temp_line.append(1)
					else:
						temp_line.append(-1)
				ret_gt[i].append(temp_line)

		ret_lt={}
		for i in range(X.shape[1]):
			ret_lt[i]=[]
			for j in range(X.shape[0]):
				temp_threshold=X[j,i]
				temp_line=[]
				for k in range(X.shape[0]):
					if X[k,i]<=temp_threshold:
						temp_line.append(1)
					else:
						temp_line.append(-1)
				ret_lt[i].append(temp_line)
		ret={}
		ret['gt']=ret_gt
		ret['lt']=ret_lt
		return ret

	#calculate error for one dimension array
	def cal_e_line(self,y,line):
		ret=0
		for i in range(len(y)):
			if y[i]!=line[i]:
				ret+=self.W[i] # 每次决策的错误率为错分样本的权重，权重越大错误率越高
		return ret

	#calculate error for two dimension array
	def cal_e_lines(self,y,lines):
		ret=[]
		for i in lines:
			ret.append(self.cal_e_line(y,i))
		return ret

	#calculate error for all possible data and get e_dic
	def cal_e_dic(self,y,dic):
		ret_gt={}
		for i in dic['gt']:
			ret_gt[i]=(self.cal_e_lines(y,dic['gt'][i]))
		ret_lt={}
		for i in dic['lt']:
			ret_lt[i]=(self.cal_e_lines(y,dic['lt'][i]))
		ret={}
		ret['gt']=ret_gt
		ret['lt']=ret_lt
		return ret

	#select min error for e_dic
	def cal_e_min(self,e_dic):
		ret=100000
		for key in e_dic:
			for i in e_dic[key]:
				temp=min(e_dic[key][i])
				if ret>temp:
					ret=temp
		for key in e_dic:
			for i in e_dic[key]:
				if ret == min(e_dic[key][i]):
					#return key,feature_index,index
					return ret,key,i,e_dic[key][i].index(ret)

	#train
	def fit(self,X,y,W):
		self.W=W
		dic=self.cal_dic(X) # 每一维特征的每个元素作为决策树阈值的分类列表，一个元素作为阈值有两种分法，大于该阈值的类分为正类1,或者大于该阈值的类分为负类-1
		e_dic=self.cal_e_dic(y,dic) # 计算所有可能分类的错误率，每个样本的错误率为其权重
		e_min,self.decision_key,self.decision_feature,e_min_i=self.cal_e_min(e_dic) # 找到最小错误率的特征位置
		self.decision_threshold=X[e_min_i,self.decision_feature]
		self.pred=dic[self.decision_key][self.decision_feature][e_min_i]
		'''
		print dic
		print e_dic
		print e_min,self.decision_key,self.decision_feature,e_min_i
		print self.decision_threshold
		print self.pred
		'''
		return

'''
X=np.array([0,12,15,23,33,46,51,72,82,100]).reshape([10,1])
y=np.array([1,1,1,-1,-1,-1,1,1,1,-1])
W=np.array([0.1]*10).reshape([10,1])/10
wk=WeakClassifier()
wk.fit(X,y,W)
'''
