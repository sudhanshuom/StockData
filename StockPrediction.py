# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:21:47 2019

@author: Sudhanshu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('stock_data.csv')

X = dataset.iloc[:,1: 7].values
Y = dataset.iloc[:,13: 19];

from sklearn.cross_validation import train_test_split

X_train1,X_test1,y_train1,y_test1 = train_test_split(X, Y, test_size = 1/5, random_state = 0)

Y1_Annual_Return = y_train1.iloc[:,0];
Y2_Excess_Return = y_train1.iloc[:,1];
Y3_Systematic_Risk = y_train1.iloc[:,2];
Y4_Total_Risk = y_train1.iloc[:,3];
Y5_Abs_Win_Rate = y_train1.iloc[:,4];
Y6_Rel_WinRate = y_train1.iloc[:,5];

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor2 = LinearRegression()
regressor3 = LinearRegression()
regressor4 = LinearRegression()
regressor5 = LinearRegression()
regressor6 = LinearRegression()

regressor1.fit(X_train1, Y1_Annual_Return)

regressor2.fit(X_train1, Y2_Excess_Return)

regressor3.fit(X_train1, Y3_Systematic_Risk)

regressor4.fit(X_train1, Y4_Total_Risk)

regressor5.fit(X_train1, Y5_Abs_Win_Rate)

regressor6.fit(X_train1, Y6_Rel_WinRate)

plt1 = plt
plt2 = plt
plt3 = plt
plt4 = plt
plt5 = plt
plt6 = plt

yy = regressor1.predict(X_test1)

plt1.scatter(X_test1[:,:1],yy,color = 'red')
plt1.plot(X_test1[:,:1],yy,color = 'blue')
plt1.title('Large B/P vs Annual Return(Training Set)')
plt1.xlabel('Large B/P')
plt1.ylabel('Annual Return')
plt1.show()

plt2.scatter(X_test1[:,1:2],yy,color = 'red')
plt2.plot(X_test1[:,1:2],yy,color = 'blue')
plt2.title('Large ROE vs Annual Return(Training Set)')
plt2.xlabel('Large ROE')
plt2.ylabel('Annual Return')
plt2.show()

plt3.scatter(X_test1[:,2:3],yy,color = 'red')
plt3.plot(X_test1[:,2:3],yy,color = 'blue')
plt3.title('Large S/P vs Annual Return(Training Set)')
plt3.xlabel('Large S/P')
plt3.ylabel('Annual Return')
plt3.show()

plt4.scatter(X_test1[:,3:4],yy,color = 'red')
plt4.plot(X_test1[:,3:4],yy,color = 'blue')
plt4.title('Large Return Rate in the last quarter vs Annual Return(Training Set)')
plt4.xlabel('Large Return Rate in the last quarter')
plt4.ylabel('Annual Return')
plt4.show()

plt5.scatter(X_test1[:,4:5],yy,color = 'red')
plt5.plot(X_test1[:,4:5],yy,color = 'blue')
plt5.title('Large Market Value vs Annual Return(Training Set)')
plt5.xlabel('Large B/P')
plt5.ylabel('Annual Return')
plt5.show()

plt6.scatter(X_test1[:,5:6],yy,color = 'red')
plt6.plot(X_test1[:,5:6],yy,color = 'blue')
plt6.title('Small systematic Risk vs Annual Return(Training Set)')
plt6.xlabel('Small systematic Risk')
plt6.ylabel('Annual Return')
plt6.show()

XX = [[0.333,0,0.333,0,0,0.333]]

print("Annual Return = ",regressor1.predict(XX))
print("Excess Return = ", regressor2.predict(XX))
print("Systematic Risk = ", regressor3.predict(XX))
print("Total Risk = ", regressor4.predict(XX))
print("Abs Win Rate = ", regressor5.predict(XX))
print("Rel WinRate = ", regressor6.predict(XX))

