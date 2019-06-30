#importing all libraries..
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing libraries from sklearn..
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os

os.chdir("C:/Users/user/Downloads/headbrain")
df=pd.read_csv("headbrain.csv")

x=df['Head Size(cm^3)'].values
y=df['Brain Weight(grams)'].values
x=x.reshape(len(x),1) #reshaping the array to 2D..

#performing train_test_split i.e dividing the data into training data to fit the regression line model and for testing the model y predictions
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
model=LinearRegression()
model.fit(x_train,y_train)
y_p=model.predict(x_test)
print(r2_score(y_test,y_p))

#data visualization..
plt.scatter(x,y)
plt.plot(x_test,y_p,color='#58b970',label='regression line')
plt.show()

#testing the algorithm by own values through predictions
print('Brain size =')
print(model.predict([[200]]))




