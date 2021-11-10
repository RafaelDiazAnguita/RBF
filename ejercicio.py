from numpy.core.fromnumeric import var
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("datasetsLA3IMC/csv/german.data-numeric.csv",header=None,delimiter='\s+')

inputs = df.values[:,0:-1]
outputs = df.values[:,-1]


scaler = MinMaxScaler()
scaler.fit_transform(inputs)
inputs = scaler.transform(inputs)


X_train, X_test, Y_train , Y_test = train_test_split(inputs,outputs,test_size=0.4,random_state=42,stratify=outputs)

KNN = KNeighborsClassifier(n_neighbors=7)
KNN.fit(X_train,Y_train)
print("KNN score: ", KNN.score(X_test,Y_test) )


LR = LogisticRegression(C=5,penalty='l1',solver='liblinear')
LR.fit(X_train,Y_train)
print("Logistic score: ", LR.score(X_test,Y_test) )


