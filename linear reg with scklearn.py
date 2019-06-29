import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import os
os.chdir("C:/Users/user/Downloads/headbrain")
df=pd.read_csv("headbrain.csv")

x=df['Head Size(cm^3)'].values
y=df['Brain Weight(grams)'].values
x=x.reshape(len(x),1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=41)
model=LinearRegression()

model.fit(x_train,y_train)
y_p=model.predict(x_test)
print(r2_score(y_test,y_p))
print(mean_squared_error(y_test,y_p))


