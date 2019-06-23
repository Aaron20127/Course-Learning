from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import pandas as pd
import warnings
import numpy as np

class Plot2D:
	
	def __init__(self,data):
		self.__data=data
		self.__X=data.iloc[:,:-1].values
		self.__le=LabelEncoder()
		self.__y=self.__le.fit_transform(data.iloc[:,-1].values)
		self.__xmin=self.__X[:,0].min()
		self.__xmax=self.__X[:,0].max()
		self.__ymin=self.__X[:,1].min()
		self.__ymax=self.__X[:,1].max()
		self.__marker_list=['o','x']

	def show(self):
		plt.xlim(self.__xmin-(self.__xmax-self.__xmin)*0.1,self.__xmax+(self.__xmax-self.__xmin)*0.1)
		plt.ylim(self.__ymin-(self.__ymax-self.__ymin)*0.1,self.__ymax+(self.__ymax-self.__ymin)*0.1)
		for i in range(2):
			temp=self.__data.loc[:][self.__data['label']==self.__le.inverse_transform([i])]
			plt.scatter(temp.iloc[:,0],temp.iloc[:,1],label=self.__le.inverse_transform(i),marker=self.__marker_list[i])
			plt.legend()
		plt.xlabel(self.__data.columns[0])
		plt.ylabel(self.__data.columns[1])
		plt.show()

	def pause(self,Seconds):
		warnings.filterwarnings("ignore",".*GUI is implemented.*")
		
		plt.ion()
		plt.xlim(self.__xmin-(self.__xmax-self.__xmin)*0.1,self.__xmax+(self.__xmax-self.__xmin)*0.1)
		plt.ylim(self.__ymin-(self.__ymax-self.__ymin)*0.1,self.__ymax+(self.__ymax-self.__ymin)*0.1)
		for i in range(2):
			class_number = np.ones(len(self.__y)).astype(np.int8) * i
			temp=self.__data.loc[:][self.__data['label']==self.__le.inverse_transform(class_number)]
			plt.scatter(temp.iloc[:,0],temp.iloc[:,1],label=self.__le.inverse_transform([i]), s = 260, marker=self.__marker_list[i])
			plt.legend(prop = {'size':18})
		plt.xlabel(self.__data.columns[0])
		plt.ylabel(self.__data.columns[1])
		plt.xticks(fontsize=20)
		plt.yticks(fontsize=20)
		plt.pause(Seconds)

'''
data=pd.read_csv('data.csv')
Plot2D(data).show()
'''