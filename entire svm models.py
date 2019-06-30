from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

df=datasets.load_iris()
def visualize():
    iris=datasets.load_iris()
    x=iris.data[:,:2]
    y=iris.target
    plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.title("sepal width and length")
    plt.show()

visualize()

def visualize1():
    iris=datasets.load_iris()
    x=iris.data[:,2:]
    y=iris.target
    plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title("petal width and length")
    plt.show()

visualize1()

#Modeling Different Kernel Svm  using Iris Sepal features
x=df.data[:,:2]
y=df.target
l_svc=svm.SVC(kernel='linear',C=1.0).fit(x,y)
l1_svc=svm.LinearSVC(C=1.0).fit(x,y)
r_svc=svm.SVC(kernel='rbf',gamma=0.7,C=1.0).fit(x,y)
p_svc=svm.SVC(kernel='poly',degree=3,C=1.0).fit(x,y)

#Visualizing the modeled svm classifiers with Iris Sepal features
x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))

title=['SVC_linear','LINEARSVC','SVC_rbf','svc_poly']
for i,clf in enumerate((l_svc,l1_svc,r_svc,p_svc)):
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(wspace=0.4,hspace=0.4)

    z=clf.predict(np.vstack([xx.ravel(),yy.ravel()]).T).reshape(xx.shape)
    plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm,alpha=0.8)
    plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title(title[i])
plt.show()

#Modeling Different Kernel Svm classifier using Iris Petal features
x=df.data[:,2:]
y=df.target
l_svc=svm.SVC(kernel='linear',C=1.0).fit(x,y)
l1_svc=svm.LinearSVC(C=1.0).fit(x,y)
r_svc=svm.SVC(kernel='rbf',gamma=0.7,C=1.0).fit(x,y)
p_svc=svm.SVC(kernel='poly',degree=3,C=1.0).fit(x,y)

#Visualizing the modeled svm classifiers with Iris petal features
x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
y_min,y_max=x[:,1].min()-1,x[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,.02),np.arange(y_min,y_max,.02))

title=['SVC_linear','LINEARSVC','SVC_rbf','svc_poly']
for i,clf in enumerate((l_svc,l1_svc,r_svc,p_svc)):
    plt.subplot(2,2,i+1)
    plt.subplots_adjust(wspace=0.4,hspace=0.4)

    z=clf.predict(np.vstack([xx.ravel(),yy.ravel()]).T).reshape(xx.shape)
    plt.contourf(xx,yy,z,cmap=plt.cm.coolwarm,alpha=0.8)
    plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.coolwarm)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.title(title[i])
plt.show()
    
