import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import os

np.random.seed(0)#for random state
os.chdir("C:/Users/user/Desktop")
df=pd.read_csv("example2.csv")

l=df['fruit'].unique()
df['fruit'].replace(l,[0,1,2,3],inplace=True)

#creating new column...
df['is_train']=np.random.uniform(0,1,len(df)) <=0.6
#then dividing into train and test datasets...
train,test=df[df['is_train']==True],df[df['is_train']==False]
features=df.columns[:2]
clf=RandomForestClassifier()
clf.fit(train[features],train['fruit'])
y_p=clf.predict(test[features])

#estimation of algorithm...
print('confusion_matrix=',confusion_matrix(test['fruit'],y_p),sep='\n')
print('accuracy_score=',accuracy_score(test['fruit'],y_p))


#tresting algorithm...
print('the output fruit for the given input data is..',l[clf.predict([[5,1]])])
