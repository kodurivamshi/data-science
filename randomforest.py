import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
os.chdir("C:/Users/user/Downloads/iris-datasets")
df=pd.read_csv("Iris.csv")

#visualiztion of sepal length and width ..wrt to species....
def visualize():
    x=df[['SepalLengthCm','SepalWidthCm']]
    y=pd.factorize(df['Species'])[0]   #it will convert the categorical data into numerical sequences as..target names and values...
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c=y,cmap=plt.cm.coolwarm)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title("sepal width and length")
    plt.show()

visualize()

#visualization of petallength and width..wrt to species..
def visualize1():
    x=df[['PetalLengthCm','PetalWidthCm']]
    y=pd.factorize(df['Species'])[0]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c=y,cmap=plt.cm.coolwarm)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title("petal width and length")
    plt.show()

visualize1()

features=df['Species'].unique()
x=df.drop('Species',axis=1)
y=pd.factorize(df['Species'])[0]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=41)
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_p=rfc.predict(x_test)

#estimation of algorithm...
print(pd.crosstab(y_test,y_p,rownames=['Actual species'],colnames=['predicted species'])) #another way of printing confusion matrix....
from sklearn.metrics import confusion_matrix,accuracy_score
print('confusion matrix',confusion_matrix(y_test,y_p),sep='\n')
print('accuracy matrix',accuracy_score(y_test,y_p))

#testing the algo...
print('outcome species for given input is...',features[rfc.predict([[1,6.0,2.2,5.0,1.5]])])
