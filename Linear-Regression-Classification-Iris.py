import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


data = pd.read_csv("iris.csv")

def matrix_inverse(X):
    return np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose())

def MSSE(X, Y):
    X_inv = matrix_inverse(X)
    W = np.matmul(X_inv, Y)
    cost = (1 / 2) * (np.matmul(Y.transpose(), Y) +np.matmul(np.matmul(X, W).transpose(),np.matmul(X, W)) - 2 * np.matmul(np.matmul(X, W).transpose(), Y))
    print("cost=",cost)
    return np.array(W)

X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
Y = data['Species']
unique_labels = np.unique(Y)

if len(unique_labels) >= 3:
    data.loc[data['Species'] == unique_labels[1], 'Species'] = unique_labels[2]

unique_labels = np.unique(Y)
print("new label's name= ",unique_labels)

def split_data(X,Y):
    X_train=pd.concat([X[:40], X[50:130]])
    Y_train = pd.concat([Y[:40], Y[50:130]])
    X_test=pd.concat([X[40:50], X[130:]])
    Y_test=pd.concat([Y[40:50], Y[130:]])
    return X_train,Y_train,X_test,Y_test

X_train,Y_train,X_test,Y_test=split_data(X,Y)
print("X_train shape",X_train.shape)
print("X_train shape",Y_train.shape)
print("X_train shape",X_test.shape)
print("X_train shape",Y_test.shape)

label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
X_train_np=np.array(X_train)
X_test_np=np.array(X_test)
Y_train_np=np.array(Y_train_encoded).reshape(-1, 1)

print("\nclassify by Sigmoid model:-")
W=MSSE(X_train_np,Y_train_np)
print("W shape= ",W.shape)
print("X_test shape= ",X_test_np.shape)

z=np.matmul(X_test_np,W)
Y3_hat=1/(1+np.exp(-z))
threshold = 0.6
classes2=np.where(Y3_hat<=threshold,unique_labels[0],unique_labels[1])
accuracy = accuracy_score(Y_test, classes2)
print("Y_hat= ",classes2)
print("accuracy of Sigmoid model: ",accuracy)

print("\nclassify by tanh model:-")
X_train = pd.concat([X[:40], X[50:130]*-1])
X_test = pd.concat([X[40:50], X[130:]*-1])
X_train_np=np.array(X_train)
X_test_np=np.array(X_test)

print("X_test after multiply second class by -1: ",X_test_np)
W=MSSE(X_train_np,Y_train_np)
print("W shape= ",W.shape)
print("X_test shape= ",X_test_np.shape)
z=np.matmul(X_test_np,W)
Y3_hat=np.tanh(z)
threshold = 0
classes2=np.where(Y3_hat<=threshold,unique_labels[0],unique_labels[1])
accuracy2 = accuracy_score(Y_test, classes2)
print("Y_hat= ",classes2)
print("accuracy of tanh model: ",accuracy2)
