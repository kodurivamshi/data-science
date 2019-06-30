from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import pandas as pd
import numpy as np

#importing inbuilt dataset iris
iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)

#creating new columns ...
df['species']=pd.Categorical.from_codes(iris.target,iris.target_names)#placing the values as in the prescribed format that [0,1,2]=[species1,species2,species3]
df['is_train']=np.random.uniform(0,1,len(df)) <= .75

#dividing data according to training set and testing set..
train,test=df[df['is_train']==True],df[df['is_train']==False]

features=df.columns[:4]
y=pd.factorize(train['species'])[0]

clf=RandomForestClassifier(random_state=0)
clf.fit(train[features],y)
y_p=clf.predict(test[features])
p=iris.target_names[y_p]

#estimating of algorithm....
#cm=pd.crosstab(test['species'],p,rownames=['autual names'],colnames=['predicted names'])
print('confusion matrix=',confusion_matrix(test['species'],p),sep='\n')
print('accuracy score=',accuracy_score(test['species'],p))

#testing with own features forn target names
print('output species for given input..',iris.target_names[clf.predict([[5.0,3.6,1.4,2.0]])])
