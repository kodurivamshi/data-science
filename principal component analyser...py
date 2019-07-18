import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
cancer=load_breast_cancer()
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


ss=StandardScaler().fit(df)
s_df=ss.transform(df)

pca=PCA(n_components=2).fit(s_df)##here actual no. of componenets=no. of features ..so we reduced it from 30 to 2 columns
r_df=pca.transform(s_df)

print(s_df.shape)
print(r_df.shape)

plt.scatter(r_df[:,0],r_df[:,1],c=cancer['target'],cmap='plasma')
##till here we reduced the dimentionallity by PCA method .now testing it with knn algo

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
np.random.seed(41)
df=pd.DataFrame(r_df)
df['outcome']=cancer.target
df['is_train']=np.random.uniform(0,1,len(df)) <=.5
train,test=df[df['is_train']==True],df[df['is_train']==False]
features=df.columns[:2]
knn=KNeighborsClassifier().fit(train[features],train['outcome'])
y_p=knn.predict(test[features])

print(accuracy_score(test['outcome'],y_p))

#testing
print(cancer.target_names[knn.predict([[-0.2345,1.2345]])])
