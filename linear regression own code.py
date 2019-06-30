#importing all required libraries

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#choosing the directory path
os.chdir("C:/Users/user/Downloads/headbrain")
df=pd.read_csv("headbrain.csv")

#capturing the independent values.
x=df["Head Size(cm^3)"].values

#capturing the dependent values.
y=df["Brain Weight(grams)"].values

#now calculating the line equation variables i.e Y=b1X+b0...(m,c)values as b1,b2
##b1=(summation(x-x_m)*(y-y_m))/summation((x-x_m)**2)....b0=Y_m-b1X_m

x_m=np.mean(x)
y_m=np.mean(y)
l=len(x)
num=denom=0
for i in range(l):
    num+=(x[i]-x_m)*(y[i]-y_m)
    denom+=(x[i]-x_m)**2
b1=num/denom
b0=y_m-b1*x_m
print(b1,b0)

#data visualizing
plt.scatter(x,y)#plotting points (x,y)
Y=b1*x+b0  #plotting the points of the linear fit regression line by line equation and independent variables
plt.plot(x,Y,color='#58b970',label='regression line')#ploting regression line
plt.show()

#estimation of algorithm
#ssr=sum of squares of reduction and sst=sum of squares of total and (sum of squares of error)sse=sst-ssr
ssr=sst=0
for i in range(l):
    Y=b0+b1*x[i]
    ssr+=(Y-y_m)**2 #ssr depends on the linear fit and 
    sst+=(y[i]-y_m)**2 #sst depends on the original dependent points
#confidential limit r^2
r2=ssr/sst
print(r2)
