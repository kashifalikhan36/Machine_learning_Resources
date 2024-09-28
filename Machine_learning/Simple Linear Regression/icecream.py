import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("IceCreamData.csv")

def lost_function(m,b,points):
    total_error=0
    for i in range(len(points)):
        x=points.iloc[i].Temperature
        y=points.iloc[i].Revenue
        total_error+=(y-(m*x+b))**2
    return total_error/float(len(points))

def gradient_descent(m_now,b_now,points,L):
    m_gradient=0
    b_gradient=0
    n=len(points)
    for i in range(n):
        x=points.iloc[i].Temperature
        y=points.iloc[i].Revenue
        m_gradient+= -(2/n)*x*(y-(m_now*x+b_now))
        b_gradient+= -(2/n)*(y-(m_now*x+b_now))
    m= m_now - m_gradient*L
    b= b_now - b_gradient*L
    return m,b

m=0
b=0
L=0.001
epochs=300

for i in range(epochs):
    m,b=gradient_descent(m,b,data,L)

print(m,b)

plt.scatter(data.Temperature,data.Revenue,color="black")
plt.plot(list(range(0,60)),[m*x+b for x in range(0,60)],color="red")
plt.show()