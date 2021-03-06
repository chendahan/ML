import requests
import pandas as pd
import re
from IPython.display import Image
from bs4 import BeautifulSoup
import os
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import datetime as dt
from sklearn.preprocessing import LabelEncoder
import math
import datetime

!pip install tmdbv3api
from tmdbv3api import TMDb
from tmdbv3api import Movie

# %matplotlib inline

"""##1 - Dataset - AirQualityUCI"""

!mkdir datasets
!mkdir datasets/AirQualityUCI
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip
!unzip AirQualityUCI.zip -d ./datasets/AirQualityUCI
df = pd.read_excel('./datasets/AirQualityUCI/AirQualityUCI.xlsx')
#drop all -200 values in order to prevent bias in the tree split, -200 is unknown value
df_filterd = df[df[df.columns[-1]] != -200] 
df_filterd.reset_index(drop=True,inplace=True)

#time is transformd using sin+cos to a numeric value (using sin+cos to maintain relations between close hours) 
def transform_time(time):
  return 0.5*math.sin(math.pi*time.hour/12)+0.5*math.cos(math.pi*time.hour/12)

for col in range(len(df_filterd.columns)):
  if(checkIfColNumeric(col,df_filterd) == False): #nominal col- treates as categorial/ordinal
    if(df_filterd.dtypes[col] == dt.time): #time- change to numeric using transform_time 
      df_filterd[df_filterd.columns[col]]=df_filterd[df_filterd.columns[col]].map(transform_time)

#always move the prediction col to be last
df_filterd=df_filterd[[c for c in df_filterd if c not in ['C6H6(GT)']] + ['C6H6(GT)']] #predict 'C6H6(GT)' - moving to last column
#train test split ratio - 70%:30%
train_data, test_data = train_test_split(df_filterd.drop(labels=['Date'],axis=1), test_size=0.3)

#for comparrison
#regression result using DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(train_data.drop(train_data.columns[-1],axis=1), train_data[train_data.columns[-1]])
print("Coefficient of determination R^2 <-- on train set: {}".format(dtr.score(train_data.drop(train_data.columns[-1],axis=1), train_data[train_data.columns[-1]])))
print("Coefficient of determination R^2 <-- on test set: {}".format(dtr. score(test_data.drop(train_data.columns[-1],axis=1), test_data[test_data.columns[-1]])))

#build tree 
#categorical_cols- list of all nominal cols (created in preprocess)
#oneHot- list of all onehot cols (created in preprocess)
my_tree = build_tree(train_data,None,None,100,LinearRegression) #data,min lines,regression model to use

#check train_data score
pred=[]
for index,row in train_data.iterrows():
  pred.append(classify(row[:-1],None,my_tree))
mse=mean_squared_error(train_data.iloc[:,-1].tolist(),pred)
print("train_data mse is:" ,mse)
r2_sc=r2_score(train_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

#check test_data score
pred=[]
for index,row in test_data.iterrows():
  pred.append(classify(row[:-1],None,my_tree))
mse=mean_squared_error(test_data.iloc[:,-1].tolist(),pred)
print("test_data mse is:" ,mse)
r2_sc=r2_score(test_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

"""##2 - Dataset - Wine Quality"""

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',delimiter=';')
df=df[[c for c in df if c not in ['density']] + ['density']] 
#train test split ratio - 70%:30%
train_data, test_data = train_test_split(df, test_size=0.3)
#for comparrison
#regression result using DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(train_data.drop(train_data.columns[-1],axis=1), train_data[train_data.columns[-1]])
print("Coefficient of determination R^2 <-- on train set: {}".format(dtr.score(train_data.drop(train_data.columns[-1],axis=1), train_data[train_data.columns[-1]])))
print("Coefficient of determination R^2 <-- on test set: {}".format(dtr. score(test_data.drop(test_data.columns[-1],axis=1), test_data[test_data.columns[-1]])))

#build tree 
#categorical_cols- list of all nominal cols (created in preprocess)
#oneHot- list of all onehot cols (created in preprocess)
my_tree = build_tree(train_data,None,None,150,LinearRegression)

#check train_data score
pred=[]
for index,row in train_data.iterrows():
  pred.append(classify(row[:-1],None,my_tree))
mse=mean_squared_error(train_data.iloc[:,-1].tolist(),pred)
print("train_data mse is:" ,mse)
r2_sc=r2_score(train_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

#check test_data score
pred=[]
for index,row in test_data.iterrows():
  pred.append(classify(row[:-1],None,my_tree))
mse=mean_squared_error(test_data.iloc[:,-1].tolist(),pred)
print("test_data mse is:" ,mse)
r2_sc=r2_score(test_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

"""##3 - Dataset - Bike Sharing """

!mkdir datasets
!mkdir datasets/bike
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip
!unzip Bike-Sharing-Dataset.zip -d ./datasets/bike

df = pd.read_csv('./datasets/bike/hour.csv')
df_filterd=df.drop(['instant','dteday'],axis=1)

#train test split ratio - 70%:30%
train_data, test_data = train_test_split(df_filterd, test_size=0.3)

#for comparrison
#regression result using DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(train_data.drop(train_data.columns[-1],axis=1), train_data[train_data.columns[-1]])
print("Coefficient of determination R^2 <-- on train set: {}".format(dtr.score(train_data.drop(train_data.columns[-1],axis=1), train_data[train_data.columns[-1]])))
print("Coefficient of determination R^2 <-- on test set: {}".format(dtr. score(test_data.drop(test_data.columns[-1],axis=1), test_data[test_data.columns[-1]])))

#categorical_cols- list of all nominal cols (created in preprocess)
#oneHot- list of all onehot cols (created in preprocess)
my_tree = build_tree(train_data,None,None,90,LinearRegression) #data,min lines,regression model to use

#check train_data score
pred=[]
for index,row in train_data.iterrows():
  pred.append(classify(row[:-1],categorical_cols,my_tree))
mse=mean_squared_error(train_data.iloc[:,-1].tolist(),pred)
print("train_data mse is:" ,mse)
r2_sc=r2_score(train_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

#check test_data score
pred=[]
for index,row in test_data.iterrows():
  pred.append(classify(row[:-1],categorical_cols,my_tree))
mse=mean_squared_error(test_data.iloc[:,-1].tolist(),pred)
print("test_data mse is:" ,mse)
r2_sc=r2_score(test_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

"""##4 - Dataset - ENB2012 Energy"""

df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx')

df_filterd=df
df_filterd.describe()

# put the class on the right
df_filterd=df_filterd[[c for c in df_filterd if c not in ['Y1']] + ['Y1']] #predict 'C6H6(GT)' - moving to last column

#train test split ratio - 70%:30%
train_data, test_data = train_test_split(df_filterd, test_size=0.3) 
#for comparrison
#regression result using DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(train_data.drop(train_data.columns[-1],axis=1), train_data[train_data.columns[-1]])
print("Coefficient of determination R^2 <-- on train set: {}".format(dtr.score(train_data.drop(train_data.columns[-1],axis=1), train_data[train_data.columns[-1]])))
print("Coefficient of determination R^2 <-- on test set: {}".format(dtr. score(test_data.drop(test_data.columns[-1],axis=1), test_data[test_data.columns[-1]])))

#categorical_cols- list of all nominal cols (created in preprocess)
#oneHot- list of all onehot cols (created in preprocess)
my_tree = build_tree(train_data,None,None,50,LinearRegression) #data,min lines,regression model to use

#check train_data score
pred=[]
for index,row in train_data.iterrows():
  pred.append(classify(row[:-1],None,my_tree))
mse=mean_squared_error(train_data.iloc[:,-1].tolist(),pred)
print("train_data mse is:" ,mse)
r2_sc=r2_score(train_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

#check test_data score
pred=[]
for index,row in test_data.iterrows():
  pred.append(classify(row[:-1],None,my_tree))
mse=mean_squared_error(test_data.iloc[:,-1].tolist(),pred)
print("test_data mse is:" ,mse)
r2_sc=r2_score(test_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

"""##5 - Dataset - Metro"""

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz')
df['date_time']=df['date_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

#pre-process categorical feature
categorical_feature_mask = df.dtypes==object
categorical_cols = df.columns[categorical_feature_mask].tolist()
le = LabelEncoder()
enc = OneHotEncoder(handle_unknown='ignore',sparse=False)
df_joind=df
for i in categorical_cols:
  x = df_joind[[i]].apply(lambda col: le.fit_transform(col))
  enc_df = enc.fit_transform(x)
  OneHot= pd.DataFrame(enc_df,columns=[x.columns[0]+str(i) for i in range(enc_df.shape[1])])
  df_joind = df_joind.join(OneHot)

#in order to differ tree split cols from the created regression onehot cols
def intersection(joindlist, dflist): 
    sub = [value for value in joindlist if value not in dflist] 
    return sub 

oneHot=intersection(df_joind.columns,df.columns)

categorical_cols # categorical_cols-list of all the categorical cols

oneHot # oneHot- list of all the columns added for the categorical variables. creates using intersection function

#time is transformd using sin+cos to a numeric value (using sin+cos to maintain relations between close hours) 
def transform_time(time):
  return 0.5*math.sin(math.pi*time.hour/12)+0.5*math.cos(math.pi*time.hour/12)

#time is transformd to the hour- cos_sin function didn't get good results 
for i in df_joind.select_dtypes(include=[np.datetime64]).columns:
  df_joind[i]=df_joind[i].map(lambda x: x.hour)

df_joind=df_joind[[c for c in df_joind if c not in ['traffic_volume']] + ['traffic_volume']] # moving to last column

# first columns are the regression columns, than the oneHot columns and the last column is the target 
df_joind

##train test split ratio - 70%:30%
train_data, test_data = train_test_split(df_joind, test_size=0.3) 

#for comparrison
#regression result using DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(train_data.drop(categorical_cols+[train_data.columns[-1]],axis=1), train_data[train_data.columns[-1]])
print("Coefficient of determination R^2 <-- on train set: {}".format(dtr.score(train_data.drop(categorical_cols+[train_data.columns[-1]],axis=1), train_data[train_data.columns[-1]])))
print("Coefficient of determination R^2 <-- on test set: {}".format(dtr. score(test_data.drop(categorical_cols+[train_data.columns[-1]],axis=1), test_data[test_data.columns[-1]])))

my_tree = build_tree(train_data,categorical_cols,oneHot,10000,LinearRegression)

#check train_data score
pred=[]
for index,row in train_data.iterrows():
  pred.append(classify(row[:-1],categorical_cols,my_tree))
mse=mean_squared_error(train_data.iloc[:,-1].tolist(),pred)
print("train_data mse is:" ,mse)
r2_sc=r2_score(train_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

#check test_data score
pred=[]
for index,row in test_data.iterrows():
  pred.append(classify(row[:-1],categorical_cols,my_tree))
mse=mean_squared_error(test_data.iloc[:,-1].tolist(),pred)
print("test_data mse is:" ,mse)
r2_sc=r2_score(test_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

"""## 6- Dataset - Parkinsons"""

df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data")

df

##train test split ratio - 70%:30%
train_data, test_data = train_test_split(df, test_size=0.3) 

#for comparrison
#regression result using DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(train_data.drop(train_data.columns[-1],axis=1), train_data[train_data.columns[-1]])
print("Coefficient of determination R^2 <-- on train set: {}".format(dtr.score(train_data.drop(train_data.columns[-1],axis=1), train_data[train_data.columns[-1]])))
print("Coefficient of determination R^2 <-- on test set: {}".format(dtr. score(test_data.drop(train_data.columns[-1],axis=1), test_data[test_data.columns[-1]])))

my_tree = build_tree(train_data,None,None,1000,LinearRegression)

#check train_data score
pred=[]
for index,row in train_data.iterrows():
  pred.append(classify(row[:-1],None,my_tree))
mse=mean_squared_error(train_data.iloc[:,-1].tolist(),pred)
print("train_data mse is:" ,mse)
r2_sc=r2_score(train_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

#check test_data score
pred=[]
for index,row in test_data.iterrows():
  pred.append(classify(row[:-1],None,my_tree))
mse=mean_squared_error(test_data.iloc[:,-1].tolist(),pred)
print("test_data mse is:" ,mse)
r2_sc=r2_score(test_data.iloc[:,-1].tolist(),pred)
print("r2_score is: ", r2_sc)

"""#Results Table"""

import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(header=dict(values=['DataSet Name','SKlearn DecisionTreeRegressor - Test Set', 'Our Tree- Test Set']),
                 cells=dict(values=[['1-AirQualityUCI','2-Wine Quality','3-Bike Sharing ','4-ENB2012 Energy','5-Metro','6-Parkinsons'],
                                    [0.9987781001800938, 0.9483771040398629, 0.9991573658586791, 0.9903379945525569,0.6250112399342129,0.6882799420490106], 
                                    [0.9999706272495746, 0.8388282078525019, 1.0, 0.9902863080283976,0.7528286997253976,0.8218897360468458]]))
                     ])
fig.show()

"""# Regression Tree Code"""

#build tree - main function
#train_data - dataFrame of the data
#the data is build in that order: first - regression cols(the data as is including the categorical cols) 
#                                 second- the one hot cols build for the categorical cols for the regression 
#                                 third- the traget col
#categorical_cols- list of all nominal cols
#oneHot- list of all onehot cols 
#numOfViews- terminate regression when num of rows is numOfViews 
#regression- which regression to use 
#   in Data-Set 5- Metro, there is a full example of how to build - train_data,categorical_cols,oneHot in the correct way when using 
#   nominal variables. when all columns are numeric, use None as categorical_cols and oneHot
def build_tree(train_data,categorical_cols,oneHot,numOfViews,regression):
    # Try to split the data
    gain, question, splits, regressionNode = find_best_split(train_data,categorical_cols,oneHot,numOfViews,regression) #split_values might be an array in case of nominal values
    # if there is no gain- return a leaf - Temp, we need to stop earlier and build regression model for the leaf 
    if gain == 0:
      return Leaf(train_data,regressionNode)

    branches=[] 
  
    # Recursively build branches.
    for i in splits:
      branches.append(build_tree(i,categorical_cols,oneHot,numOfViews,regression))

    return Decision_Node(question,branches,regressionNode)


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float) or isinstance(value, np.integer)

class Question: #node 

    def __init__(self, column, value): #value is an array in case of nominal val
        self.column = column
        self.value = value

    def match(self, row): 
      val = row[self.column]
      if is_numeric(val):
        return val <= self.value
      else: 
        j=0
        for i in self.value:
          if val == i:
            return j
          j+=1
        return None

    def is_numeric_(self):
      return isinstance(self.value, int) or isinstance(self.value, float) or np.issubdtype(value, np.integer)

    def __repr__(self): #print
        condition = "=="
        if is_numeric_(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))
        
        
class Decision_Node:
    def __init__(self,
                 question,
                 branches,
                 regressionNode):
        self.question = question
        self.branches = branches
        self.regressionNode = regressionNode #regression for nominal values, used for cases we get unknown value during classification . Null for numeric


class Leaf:
  def __init__(self, data,regression):
    self.regression = regression
    if(regression==None):
      self.prediction_mean = data[data.columns[-1]].mean()
  
  def prediction(self,row):
    if (self.regression != None):
      return self.regression.predict(row.values.reshape(1,-1))[0]
    else:
      return self.prediction_mean



def handleNominal(col,train_data,curr_mse):
  gain=0
  splits=[]
  #for nominal value: split for each unique value
  values=set([row[col] for i,row in train_data.iterrows()])
  values = list(values)
  question = Question(train_data.columns[col], values)
  for i in values:
    splits.append(train_data[train_data[train_data.columns[col]] == i])
  summse=0
  for i in splits:
    summse+= mean_squared_error(i.iloc[:,-1].tolist(),np.full(len(i),i[i.columns[-1]].mean()))*(len(i)/len(train_data))
  gain=curr_mse-summse
  return gain, question,splits 


def handleNumeric(col,train_data,curr_mse):
  best_gain=0
  part1best=[]
  part2best=[]
  splits=[]
  best_question=None
  #for numerical value: choose one split point 
  sorted_train_data = train_data.sort_values(by=train_data.columns[col],axis=0,ascending=True)  #sort according to col
  last_index=0
  arr_len=len(sorted_train_data)
  train_data_unique=train_data[train_data.columns[col]].unique()
  train_data_unique.sort()
  for i in train_data_unique:
    while (last_index<arr_len and sorted_train_data.iloc[last_index][col] <= i ):
      last_index+=1
    if(last_index<arr_len):
      question = Question(train_data.columns[col], i)
      part1=sorted_train_data.iloc[:last_index,]
      part2=sorted_train_data.iloc[last_index:,]
      if len(part1) == 0 or len(part2) == 0:
        continue
      part1_mse=mean_squared_error(part1.iloc[:,-1].tolist(),np.full(len(part1),part1[part1.columns[-1]].mean()))
      part2_mse=mean_squared_error(part2.iloc[:,-1].tolist(),np.full(len(part2),part2[part2.columns[-1]].mean()))
      all_mse= part1_mse*len(part1)/len(sorted_train_data) + part2_mse*len(part2)/len(sorted_train_data)
      gain=curr_mse-all_mse
      if gain > best_gain:
        best_gain=gain
        best_question=question
        part1best=part1
        part2best=part2
  splits.append(part1best)
  splits.append(part2best)
  return best_gain, best_question,splits 



def checkIfColNumeric(col,train_data):
  for i in range(len(train_data)):
    if(train_data.iloc[i,col]!=None and is_numeric(train_data.iloc[i,col])):
      return True
  return False


#find best feature to split with and best value 
def find_best_split(train_data,categorical_cols,oneHot,numOfViews,regression):
    best_gain = 0 
    best_question = None
    question=None
    part1best=None
    part2best=None
    regressionNode=None
    bestregressionNode=None
    reg=False
    splits=[]

    if len(train_data) <= numOfViews: 
      regressionNode=regression()
      if(categorical_cols):
        regressionNode.fit(train_data.drop(categorical_cols+[train_data.columns[-1]], axis=1),train_data[list(train_data.columns)[-1]]) 
      else:
        regressionNode.fit(train_data.drop(train_data.columns[-1], axis=1),train_data[list(train_data.columns)[-1]]) 
      return best_gain, best_question,splits,regressionNode

    curr_mse=mean_squared_error(train_data.iloc[:,-1].tolist(),np.full(len(train_data),train_data[train_data.columns[-1]].mean()))
    n_features = len(train_data.columns)-1
    if(oneHot):
      n_features=n_features-len(oneHot)# number of columns minus the label and the oneHot
    for col in range(n_features):  # for each feature
      reg=False
      if(checkIfColNumeric(col,train_data)):
        best_gain_temp, best_question_temp,splits_temp =handleNumeric(col,train_data,curr_mse)
      else:
        best_gain_temp, best_question_temp,splits_temp =handleNominal(col,train_data,curr_mse)
        reg=True
        #create regression for nominal values - used for cases we get unknown value during classification 
        regressionNode=regression()
        if(categorical_cols):
          regressionNode.fit(train_data.drop(categorical_cols+[train_data.columns[-1]], axis=1),train_data[list(train_data.columns)[-1]]) 
        else:
          regressionNode.fit(train_data.drop(train_data.columns[-1], axis=1),train_data[list(train_data.columns)[-1]]) 
      if best_gain_temp > best_gain:
        best_gain=best_gain_temp
        best_question=best_question_temp
        splits=splits_temp
        if (reg == True):
          bestregressionNode=regressionNode
        else:
          bestregressionNode=None

    return best_gain, best_question,splits,bestregressionNode

"""# Classify Function 

"""

#row- row to classify
#categorical_cols- row that need to be removed during regression. (categorical_cols same as in train )
def classify(row,categorical_cols, node):
  
  if isinstance(node, Leaf):
    if(categorical_cols):
      return node.prediction(row.drop(categorical_cols))
    else:
      return node.prediction(row)

  a=node.question.match(row)

  if isinstance(a,(np.bool,np.bool_,np.bool8)): #numeric
    if a==True:
      return classify(row,categorical_cols, node.branches[0])
    else:
      return classify(row,categorical_cols, node.branches[1])
  else: #nominal
    if(a == None): # no match
      if(categorical_cols):
        return node.regressionNode.predict(row.drop(categorical_cols).values.reshape(1,-1))[0] 
      else:
        return node.regressionNode.predict(row.values.reshape(1,-1))[0] 
    else:
      return classify(row,categorical_cols, node.branches[a])
