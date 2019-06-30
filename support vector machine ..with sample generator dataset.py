import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets.samples_generator import make_blobs
##make_blobs from saples_generator is for testing our algo by creating a sample dataset

x,y=make_blobs(n_samples=40,centers=2,random_state=20)
clf=svm.SVC(kernel='linear',C=1)
clf.fit(x,y)
plt.scatter(x[:,0],x[:,1],c=y,s=30,cmap=plt.cm.Paired)
#plt.show()

if clf.predict([[3,4]])==0:
    print('alligator')
else:
    print("crocodile")

#data visualization of hyperplane...

#plot the decision function
ax=plt.gca() ##inorder to get the scale of axis of above ploted graph
xlim=ax.get_xlim()
ylim=ax.get_ylim()


##create grid to evaluate model
xx=np.linspace(xlim[0],xlim[1],30)
yy=np.linspace(ylim[0],ylim[1],30)
##by mesgrid it froms YY,XX as (30,30) 2-D array..
YY,XX=np.meshgrid(yy,xx)#it creates an array of (x,y)points for rectangular shape grid points in graph as (30,30) for both..
xy=np.vstack([XX.ravel(),YY.ravel()]).T##ravel() means it concate the all the rows in an array..
##numpy.vstack() function is used to stack the sequence of input arrays vertically to make a single array.

z=clf.decision_function(xy).reshape(XX.shape)
##decide weather to plot the points around hyperpalne i.e one vs one or one vxsrest

##plot decision boundary(hyperpalne) and margins..
ax.contour(XX,YY,z,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
#plot support vextors
#ax.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=100,linewidth=1,facecolors='none')
plt.scatter(3,4,s=100,color='black')##testing of above point predicted..
plt.show()



