import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, e
import math


df_train = pd.read_csv("/Users/Neil/desktop/university/courses/cmpt459/Assignment_1_Datasets/banks.csv")
df_test = pd.read_csv("/Users/Neil/desktop/university/courses/cmpt459/Assignment_1_Datasets/banks-test.csv")
df_interview = pd.read_csv("/Users/Neil/desktop/university/courses/cmpt459/Assignment_1_Datasets/interviewee.csv")

# loading data
# printing data
# print(df_train)
# print(df_test)
# print(df_interview)

# split data into input and taget variable
# x = df_interview.drop("label", axis = 1)
# y = df_interview["label"]


def cal_entropy(target_col):
  """ Computes entropy of label distribution. """

  numOfLabels = len(target_col)

  if numOfLabels <= 1:
    return 0

  elements,counts = np.unique(target_col, return_counts=True)
  p = counts / numOfLabels

  numOfClasses = np.count_nonzero(p)

  if numOfClasses <= 1:
    return 0

  entropy = np.sum([-(i * np.log2(i)) for i in p])

  return entropy


# print(cal_entropy(df_train["label"]))

def informationGain(data, feature_col, target_col):
	numOfLabels = len(feature_col)

	if numOfLabels <= 1:
		return 0

	elements,counts = np.unique(data[feature_col], return_counts=True)
	p = counts / numOfLabels

	numOfClasses = np.count_nonzero(p)

	if numOfClasses <=1:
		return 0

	weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * cal_entropy((data[data[feature_col]==elements[i]]).dropna()[target_col]) for i in range(len(elements))])


	info_gain = cal_entropy(data[target_col]) - weighted_entropy

	return info_gain

#print(informationGain(df_train,"region","label"))


def best_split(data, features, label):
	item_values = [informationGain(data ,feature, label) for feature in features]
	index = np.argmax(item_values)
	best_feature = features[index]
	return best_feature


def decision_tree(data, data2, features, label, parent = None):

	datum = np.unique(data2[label], return_counts=True)

	if(len(np.unique(data[label])) <= 1):
		return np.unique(data[label])[0]

	elif(len(data) == 0):
		return np.unique(data[label])[np.argmax(np.unique(data2[label], return_counts=True)[1])]

	elif(len(features) == 0):
		return parent

	else:
		
		parent = np.unique(data[label])[np.argmax(np.unique(data2[label], return_counts=True)[1])]

	best_feature = best_split(data,features,label)

	decisionTree = {best_feature:{}}

	features.remove(best_feature)

	for value in np.unique(data[best_feature]):
		min_data = data.where(data[best_feature] == value).dropna()
		sub_tree = decision_tree(min_data, data2, features, label, parent)
		decisionTree[best_feature][value] = sub_tree

	return decisionTree


def get_prediction(x_dict,decisionTree,default =1):
	for key in list(x_dict.keys()):
		if key in list(decisionTree.keys()):
			try:
				result = decisionTree[key][x_dict[key]]
				if isinstance(result, dict):
					return get_prediction(x_dict,result)
				else:
					return result
			except:
				return default

		

def predict(data, decisionTree):
	x_dict = data.iloc[:,:-1].to_dict(orient = "records")
	predictions = pd.DataFrame(columns=["Predictions"])
	outf = open("/Users/Neil/desktop/university/courses/cmpt459/Assignment_1_Datasets/Predictions.csv", "w+")
	for i in range(len(data)):
		predictions.loc[i,"Predictions"] = get_prediction(x_dict[i], decisionTree, 1.0)
		outf.write(str(predictions) + '\n')

	print('The prediction accuracy is: ',(np.sum(predictions["Predictions"] == data["label"])/len(data))*100,'%')


def RandomForest(numberOfTrees, percetageOfAttributes):
	forest = []
	for i in range(numberOfTrees):
		sampleTrainDf = df_train.sample(frac=percetageOfAttributes)
		decisionTree = decision_tree(sampleTrainDf, sampleTrainDf, sampleTrainDf.columns[:-1].tolist(), 'label')
		sampleTestDf = df_test.sample(frac=percetageOfAttributes)
		decisionTree = decision_tree(sampleTrainDf, sampleTrainDf, sampleTestDf.columns[:-1].tolist(), 'label')
		forest.append(decisionTree)
	return forest

def randomForestPredictions(data, randomForest):
	predictions = {}

	prediction = pd.DataFrame(columns=["Predictions"])
	for i in range(len(randomForest)):
		column = "decision tree " + str(i)
		prediction.loc[column,"Predictions"] = predict(data, randomForest[i])	
	return prediction


# tree = decision_tree(df_test, df_test, df_test.columns[:-1].tolist(),'label')
# predict(df_test,tree)
# 
for i in range(10):
	print("No. of Trees: " + str(5*i) + "Percetage Of Attributes: " + str(0.1*i*100))
	f = RandomForest((5*i),(0.1*i))
	randomForestPredictions(df_test, f)



