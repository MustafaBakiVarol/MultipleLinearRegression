# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 15:40:47 2023

@author: asus
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression

df = pd.read_csv("multiple_linear_regression_dataset.csv" , sep = ";")

x= df.iloc[:,[0,2]].values
y= df.maas.values.reshape(-1,1)

multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(x,y)

print("b0:" , multiple_linear_reg.intercept_)
print("b1 , b2 :" , multiple_linear_reg.coef_)

multiple_linear_reg.predict(np.array([[10,35], [5,35]]))