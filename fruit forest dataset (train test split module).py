import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import os


os.chdir("C:/Users/user/Desktop")
df=pd.read_csv("example2.csv") ##here I have saved the fruit dataset file as example2.. 
l=df['fruit'].unique()
df['fruit'].replace(l,[0,1,2,3],inplace=True)

#train test split...
x=df.drop('fruit',axis=1)
y=df['fruit']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
y_p=clf.predict(x_test)

#estimation of algorithm...
print('confusion_matrix=',confusion_matrix(y_test,y_p),sep='\n')
print('accuracy_score=',accuracy_score(y_test,y_p))

#testing the algorithm...
print('the output fruit for the given input data is..',l[clf.predict([[1,5]])])
