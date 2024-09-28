import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("IceCreamData.csv")

x=data[["Temperature"]]
y=data[["Revenue"]]
X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict([[300]])

print(regressor.intercept_)
print(regressor.coef_)
print(y_pred)