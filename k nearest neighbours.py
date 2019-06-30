#importing libraries...
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
import os

os.chdir("C:/Users/user/Downloads/diabetes")
df=pd.read_csv("diabetes.csv")
print(df.head())

#now cleaning the dataset from zeros....
Zeros_not_accepted=['Glucose','BloodPressure','BMI','Insulin','SkinThickness']
#here we are replacing the zeors in above zero_not_accepted columns with mean of there columns values..
for col in Zeros_not_accepted:
    df[col]=df[col].replace(0,np.NaN)
    mean=int(df[col].mean(skipna=True))
    df[col]=df[col].replace(np.NaN,mean)
print(df)

##train test split .....\
x=df.iloc[:,0:8]
y=df.iloc[:,8]
#standardizing the x values to mean=0 and variance=1...
ss=StandardScaler().fit(x)
s_x=ss.transform(x)

x_train,x_test,y_train,y_test=train_test_split(s_x,y,test_size=0.2,random_state=0)
#fitting the data
#inorder to find the distance between new point and existing point use n_neighbors=math.sqrt(len(y_test))..if it is even then -1 and metric=euclidean..
model=KNeighborsClassifier().fit(x_train,y_train)
y_p=model.predict(x_test)

#estimating the algorithm...
print('confusion matrix',confusion_matrix(y_test,y_p),sep='\n')
print('accuracy_score=',accuracy_score(y_test,y_p))

#testing the algorithm...
print('the output for given input is ..',model.predict([[-0.5153943 ,  0.03377495, -2.04116806,  0.30215916,  0.14450502,
        1.31580706,  0.15169799, -0.60173245]]))
