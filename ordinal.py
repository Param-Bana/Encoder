import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
df=pd.read_csv("customer.csv")
df=df.iloc[:,2:]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,0:2],df.iloc[:,-1],test_size=0.25,random_state=0)
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[
    ['Poor', 'Average', 'Good', 'Very Good', 'Excellent'],
    ['School', 'UG', 'PG', 'PhD']
])
oe.fit(X_train)
X_encoded_train=oe.transform(X_train)
X_encoded_test=oe.transform(X_test)
#above data is numpy array, we need to convert it into dataframe
X_encoded_train=pd.DataFrame(X_encoded_train,columns=X_train.columns)
X_encoded_test=pd.DataFrame(X_encoded_test,columns=X_test.columns)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y_train)
y_encode_train=le.transform(y_train)
y_encode_test=le.transform(y_test)
y_encode_train=pd.DataFrame(y_encode_train,columns=['Purchased'])
print(y_encode_train)
