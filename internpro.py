import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("sssbankdata1.csv")
print df.head()

#checking whether any null data is available or not
dt=df.isnull().sum()
print dt


#visualizaion of age 
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(df['age'],bins=7,color='red')
plt.title('age distribution')
plt.xlabel('age')
plt.ylabel('total count')
plt.show()

#income
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(df['income'],bins=7,color='skyblue')
plt.title('income  distribution')

plt.xlabel('income per annum')
plt.ylabel('total count')
plt.show()


df['other assets'] = df['other assets'].apply(lambda x: x*100000)

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(df['other assets'],bins=7,color='yellow')
plt.title("customer's assets  distribution")

plt.xlabel('assets value')
plt.ylabel('total people count')
plt.show()


df['house value'] = df['house value'].apply(lambda x: x*100000)
print df.head()
fig=plt.figure()
at=fig.add_subplot(1,1,1)
at.hist(df['house value'],bins=7,color='orange')
plt.title('house weightage  distribution')
plt.xlabel('house value in lakhs')
plt.ylabel('total count')
plt.show()


df['lib'] = df['lib'].apply(lambda x: x*100000)
print df.head()


# Percentage calculation
df['40inc'] = df['income'].apply(lambda x: x*.4)
print df.head()


#calculation
df["total asset"]=df['40inc']+df['other assets']+df['house value']
print df.head()

df['update'] = df['if yes value'].apply(lambda x: x*100000)

#finding the person worth

df['net worth']=df['total asset']-df['lib']-df['update']
print df.head()

#making the workclass and visualizing
uniq=df['workclass'].unique()
print uniq

df['workclass']=df['workclass'].replace({
'Private':0, 'Local-gov':1, '?':2, 'Self-emp-not-inc':3, 'Federal-gov':4, 'State-gov':5,
 'Self-emp-inc':6})
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(df['workclass'],bins=7,color='green')
plt.title("workclass differentiation")
plt.text(4, 650, '0-private')
plt.text(4, 630, '1-local government')
plt.text(4, 610, '2-?')
plt.text(4, 600, '3- self employee without inc')
plt.text(4, 580, '4- federal government')
plt.text(4, 560, '5-sate government')
plt.text(4, 540, '6-self employee with inc')

plt.xlabel('different workplace')
plt.ylabel('total people count')
plt.show()
data=pd.read_csv('abc.csv')
"""x=df[['age']].values
y=df[['worth']].values
#fitting polynomial regression to the dataset
#fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import  LinearRegression

poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
poly_reg=LinearRegression()
poly_reg.fit(x_poly,y)

plt.plot(x_poly,y)
plt.show()

print (x_poly)"""

X = df.iloc[:, 2:3].values 
y = df.iloc[:, 10].values 

print X
print y

# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression() 
  
lin.fit(X, y) 

# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression() 
  
lin.fit(X, y) 

# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 
  
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 

# Visualising the Linear Regression results 
plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('age') 
plt.ylabel('worth') 
  
plt.show() 

# Visualising the Polynomial Regression results 
plt.scatter(X, y, color = 'blue') 
  
plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('age') 
plt.ylabel('worth') 
  
plt.show() 

# Predicting a new result with Linear Regression 
t=lin.predict(110.0) 
print ("after training the new result of linear regression")
print t
# Predicting a new result with Polynomial Regression 
q=lin2.predict(poly.fit_transform(110.0)) 

print ("after training the new result of polynomial regression")

print q



#decision tree



# select all rows by : and column 1 
# by 1:2 representing features 
#X = df[:, 1:2].astype(int)  
  
# print X 
#print(X) 
X = df.iloc[:, 2:3].values 
y = df.iloc[:, 13].values 
print X
# select all rows by : and column 2 
# by 2 to Y representing labels 
#y = df[:, 13].astype(int)  
  
# print y 
print(y) 

# import the regressor 
from sklearn.tree import DecisionTreeRegressor  
  
# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 0)  
  
# fit the regressor with X and Y data 
regressor.fit(X, y) 

# predicting a new value 
  
# test the output by changing values, like 3750 
y_pred = regressor.predict(1000) 
  
# print the predicted price 
print("Predicted price: % d\n"% y_pred)  

# arange for creating a range of values  
# from min value of X to max value of X  
# with a difference of 0.01 between two 
# consecutive values 
X_grid = np.arange(min(X), max(X), 0.01) 
  
# reshape for reshaping the data into  
# a len(X_grid)*1 array, i.e. to make 
# a column out of the X_grid values 
X_grid = X_grid.reshape((len(X_grid), 1))  
  
# scatter plot for original data 
plt.scatter(X, y, color = 'red') 
  
# plot predicted data 
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')  
  
# specify title 
plt.title('predictig persons eligibility (Decision Tree Regression)')  
  
# specify X axis label 
plt.xlabel('net worth') 
  
# specify Y axis label 
plt.ylabel('age') 
  
# show the plot 
plt.show() 

# import export_graphviz 
from sklearn.tree import export_graphviz  
  
# export the decision tree to a tree.dot file 
# for visualizing the plot easily anywhere 
export_graphviz(regressor, out_file ='treeq.dot', 
               feature_names =['net worth'])  

import pydotplus 
from sklearn import tree
import collections

#data collection
#x=[[180,15,0],[177,42,0],[136,35,1],[174,65,0],[141,28,1]]
#y=['man','women','women','man','woman']
X = df.iloc[:, 2:3].values 
y = df.iloc[:, 13].values 

data_feature_names =["height","hair length",'voice pitch']

#training

clf=tree.DecisionTreeClassifier()
clf=clf.fit(X,y)

#visualization data
dot_data=tree.export_graphviz(clf,feature_names=data_feature_names,out_file=None,filled=True,rounded=True)
graph=pydotplus.graph_from_dot_data(dot_data)

colors=("red","orange")
edges=collections.defaultdict(list)

for edge in graph.get_edge_list():
	edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
	edges[edge].sort()
	for i in range(2):
		dest=graph.get_node(str(edges[edge][i]))[0]
		dest.set_fillcolor(colors[i])

graph.write_png('tre.dot')
