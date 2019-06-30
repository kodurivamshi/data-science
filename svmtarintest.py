import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import os
os.chdir("D:/[FreeTutorials.Eu] Udemy - Python for Data Science and Machine Learning Bootcamp")
df=pd.read_csv("Book1.csv")##using  the dataset of cupcake and muffin dishes.preparation ingredints..
sns.lmplot('sugar','butter',data=df,hue='type',palette='coolwarm',fit_reg=False,scatter_kws={"s":70})
l=df['type'].unique()
df['type'].replace(l,[0,1],inplace=True)
x=df[['sugar','butter']].as_matrix()
y=df['type'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=41)
model=svm.SVC(kernel='linear',C=1)
model.fit(x_train,y_train)
y_p=model.predict(x_test)
#testing...
print(l[model.predict([[12,12]])])
#estimating...
print(accuracy_score(y_test,y_p))

#plotting dataset..
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,s=30,cmap=plt.cm.Paired)

#plot the decision function...
ax=plt.gca()##inorder to get the scale of axis of above ploted graph
xlim=ax.get_xlim()
ylim=ax.get_ylim()

##create grid to evaluate model
xx=np.linspace(xlim[0],xlim[1],30)
yy=np.linspace(ylim[0],ylim[1],30)

##by mesgrid it froms YY,XX as (30,30) 2-D array..
YY,XX=np.meshgrid(yy,xx)#it creates an array of (x,y)points for rectangular shape grid points in graph as (30,30) for both..
xy=np.vstack([XX.ravel(),YY.ravel()]).T#ravel() means it concate the all the rows in an array..
##numpy.vstack() function is used to stack the sequence of input arrays vertically to make a single array.

z=model.decision_function(xy).reshape(XX.shape)
##decide weather to plot the points around hyperpalne i.e one vs one or one vxsrest

##plot decision boundary(hyperpalne) and margins..
ax.contour(XX,YY,z,colors='k',levels=[-1,0,1],linestyles=['--','-','--'])
plt.scatter(12,12,s=100,color='black')##testing of above point predicted..
plt.show()
