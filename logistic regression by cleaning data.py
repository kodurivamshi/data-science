##importing libraries....
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#import libraries from sklearn...
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import os

os.chdir("C:/Users/user/Downloads/titanic-data")
df=pd.read_csv("titanic.csv")
print(df.head())

##checking for NAN values and visualizing too.for NAN values and then cleaning it
print(df.isnull().sum())
sb.heatmap(df.isnull())
plt.show()

df.drop('Cabin',axis=1,inplace=True)
df.dropna(inplace=True)
print(df.isnull().sum())

##creating dummmies for categorical data which is in string format
dsex=pd.get_dummies(df['Sex'])
dsex.drop('female',axis=1,inplace=True)
dem=pd.get_dummies(df['Embarked'],drop_first=True)
#conating the original dataframe with dummies set....
df=pd.concat([df,dsex,dem],axis=1)


##remvoing the real dummies data and some object type data that consists of string type
df.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)

##training and testing data
x=df.drop('Survived',axis=1)
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
model=LogisticRegression()
model.fit(x_train,y_train)
y_p=model.predict(x_test)


#estimation of algorithm...
print(classification_report(y_test,y_p))
print('confusion_matrix =',confusion_matrix(y_test,y_p),sep='\n')
print('accuracy_score=',accuracy_score(y_test,y_p))

#testing the algorithm...by own input..set
print('the given person for testing is ',model.predict([[3,31.0,1,0,18.0000,0,0,1]]))
##tetsing the algo if 0=not survived and if 1=survived
