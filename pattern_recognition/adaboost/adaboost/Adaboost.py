
from __future__ import division
import numpy as np 
from WeakClassifier import *

class AdaboostClassifier:

	#calculate new Weight, and normalize the weight
	def cal_W(self,e,W,alpha,y,pred):
		new_W=[]
		for i in range(len(y)):
			# 原文绝对值处不需要归一化，因为标签在是0和1；而这里的标签是-1和1。
			# 所以需要在绝对值处除以2，将值归一化
			# new_W.append(W[i] * (e/(1-e))**((1-np.abs(y[i] - pred[i])/2.0))) 
			# new_W.append(W[i]*np.exp(-0.5*alpha*y[i]*pred[i]))
			# adaboost M1
			if (y[i] == pred[i]): 
				new_W.append(W[i])
			else:
				new_W.append(W[i]*np.exp(alpha))
		
		return np.array(new_W/sum(new_W)).reshape([len(y),1])

	#calculate error rate per iteration
	def cal_e(self,y,pred,W):
		ret=0
		for i in range(len(y)):
			if y[i]!=pred[i]:
				ret+=W[i]
		return ret

	#calculate alpha
	def cal_alpha(self,e):
		if e==0:
			return 10000
		elif e==0.5:
			return 0.001
		else:
			return np.log((1-e)/e) # log表示ln，e=0.5时值为0， e=0时值为无穷大，所以要做前边两个判断
			# return e

	#calculate final predict value
	def cal_final_pred(self,i,alpha,weak,y):
		ret=np.array([0.0]*len(y))
		alpha_total = 0.0
		for j in range(i+1):
			ret+=alpha[j]*weak[j].pred
			alpha_total += alpha[j]

		# 将输出归一化，这里由于决策阈值为0，归一化意义不大。但如果输出样本的标签为0-1，
		# 则决策的阈值为1/2时，就必须将结果归一化
		# return np.sign(ret)
		normalized_output = ret/alpha_total 
		return np.sign(normalized_output) 

	#calculate final error rate
	def cal_final_e(self,y,cal_final_predict):	
		ret=0
		for i in range(len(y)):
			if y[i]!=cal_final_predict[i]:
				ret+=1
		return ret/len(y) # 最后一次错误率使用的是错误样本数占总样本的比例，并不是权值

	#train
	def fit(self,X,y,M=1500):
		W={}
		self.weak={}
		alpha={}
		pred={}

		for i in range(M):
			W.setdefault(i)
			self.weak.setdefault(i)
			alpha.setdefault(i)
			pred.setdefault(i)

		#per iteration (all:M times)
		for i in range(M):
			#for the first iteration,initial W
			if i == 0:
				W[i]=np.array([1]*len(y))/len(y)
				W[i]=W[i].reshape([len(y),1])
			#if not the first iteration,calculate new Weight
			else:
				W[i]=self.cal_W(e, W[i-1],alpha[i-1],y,pred[i-1])

			# print ('W=%s'%(W[i].reshape(35)))
			# if i > 0: print ('pred=%s'%(np.array(pred[i-1]).reshape(35)))
			# print ('y   =%s'%(y))
	
			#using train weak learner and get this learner predict value
			self.weak[i]=WeakClassifier()
			self.weak[i].fit(X,y,W[i])
			pred[i]=self.weak[i].pred

			#calculate error rate this iteration
			e=self.cal_e(y,pred[i],W[i])
			#calculate alpha this iteration
			alpha[i]=self.cal_alpha(e)
			#calculate the final predict value
			cal_final_predict=self.cal_final_pred(i,alpha,self.weak,y)

			print ('iteration:%d'%(i+1))
			print ('self.decision_key=%s'%(self.weak[i].decision_key))
			print ('self.decision_feature=%d'%(self.weak[i].decision_feature))
			print ('decision_threshold=%f'%(self.weak[i].decision_threshold))
			print ('W=%s'%(W[i]))
			print ('pred=%s'%(pred[i]))
			print ('e:%f alpha:%f'%(e,alpha[i]))
			print ('cal_final_predict:%s'%(cal_final_predict))
			print ('cal_final_e:%s%%'%(self.cal_final_e(y,cal_final_predict)*100))
			print ('')

			#calculate the final error rate,if it is zero,stop iteration.
			if self.cal_final_e(y,cal_final_predict)==0 or e==0:
				break
		#return the iteration times,from 1 on.
		return i+1

