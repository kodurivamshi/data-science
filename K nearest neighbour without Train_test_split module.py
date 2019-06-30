#importing libraries...
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os

#for random state..
np.random.seed(0)
os.chdir("C:/Users/user/Downloads/diabetes")
df=pd.read_csv("diabetes.csv")
print(df.head())

#creating one more column for train and testing u r dataset..with inbuilt libraries i.e train_test_split
df['is_train']=np.random.uniform(0,1,len(df)) <=.5
print(df.head())

#now cleaning the dataset from zeros....
Zeros_not_accepted=['Glucose','BloodPressure','BMI','Insulin','SkinThickness']
#here we are replacing the zeors in above zero_not_accepted columns with mean of there columns values..
for col in Zeros_not_accepted:
    df[col]=df[col].replace(0,np.NaN)
    mean=int(df[col].mean(skipna=True))
    df[col]=df[col].replace(np.NaN,mean)
print(df)

#now dividing the data into train,test....
train,test=df[df['is_train']==True],df[df['is_train']==False]

features=df.columns[:8]

knn=KNeighborsClassifier()
knn.fit(train[features],train['Outcome'])
y_p=knn.predict(test[features])

#estimating the data..
print('confusion_matrix',confusion_matrix(test['Outcome'],y_p),sep='\n')
print('accuracy_score=',accuracy_score(test['Outcome'],y_p))

#testing own target by giving features...
print('outcome of given input is ',knn.predict([[-0.5153943 ,  0.03377495, -2.04116806,  0.30215916,  0.14450502,
        1.31580706,  0.15169799, -0.60173245]]))
