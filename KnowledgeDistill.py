import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
T = 1000                   #temperature

data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

def init_params():
    W1 = np.random.rand(40, 784) - 0.5 
    b1 = np.random.rand(40, 1) - 0.5
    W2 = np.random.rand(40, 40) - 0.5
    b2 = np.random.rand(40, 1) - 0.5
    W3 = np.random.rand(10, 40) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 50 == 0:
            print("Iteration ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A3), Y))
    return W1, b1, W2, b2, W3, b3

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    return get_predictions(A3)

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    

# TW1, Tb1, TW2, Tb2, TW3, Tb3 = gradient_descent(X_train, Y_train, 500, 0.1)
TW1, Tb1, TW2, Tb2, TW3, Tb3 = [], [], [], [], [], []

# with open("teacher.csv", "w") as file:
#     for layer in TW1:
#         file.write(str(layer[0]))
#         for value in layer[1:]:
#             file.write(",")
#             file.write(str(value))
#         file.write("/")
#     file.write("\n")
    
#     for layer in Tb1:
#         file.write(str(layer[0]))
#         for value in layer[1:]:
#             file.write(",")
#             file.write(str(value))
#         file.write("/")
#     file.write("\n")

#     for layer in TW2:
#         file.write(str(layer[0]))
#         for value in layer[1:]:
#             file.write(",")
#             file.write(str(value))
#         file.write("/")
#     file.write("\n")

#     for layer in Tb2:
#         file.write(str(layer[0]))
#         for value in layer[1:]:
#             file.write(",")
#             file.write(str(value))
#         file.write("/")
#     file.write("\n")

#     for layer in TW3:
#         file.write(str(layer[0]))
#         for value in layer[1:]:
#             file.write(",")
#             file.write(str(value))
#         file.write("/")
#     file.write("\n")

#     for layer in Tb3:
#         file.write(str(layer[0]))
#         for value in layer[1:]:
#             file.write(",")
#             file.write(str(value))
#         file.write("/")
#     file.write("\n")


with open("teacher.csv", "r") as file:
    line = file.readline()
    line = line.replace("\n", "")
    line = line.split("/")
    line = line[:-1]
    i = 0
    for layer in line:
        TW1.append([])
        layer = layer.split(",")
        for value in layer:
            TW1[i].append(float(value))
        i += 1

    line = file.readline()
    line = line.replace("\n", "")
    line = line.split("/")
    line = line[:-1]
    i = 0
    for layer in line:
        Tb1.append([])
        layer = layer.split(",")
        for value in layer:
            Tb1[i].append(float(value))
        i += 1
    
    line = file.readline()
    line = line.replace("\n", "")
    line = line.split("/")
    line = line[:-1]
    i = 0
    for layer in line:
        TW2.append([])
        layer = layer.split(",")
        for value in layer:
            TW2[i].append(float(value))
        i += 1
    
    line = file.readline()
    line = line.replace("\n", "")
    line = line.split("/")
    line = line[:-1]
    i = 0
    for layer in line:
        Tb2.append([])
        layer = layer.split(",")
        for value in layer:
            Tb2[i].append(float(value))
        i += 1

    line = file.readline()
    line = line.replace("\n", "")
    line = line.split("/")
    line = line[:-1]
    i = 0
    for layer in line:
        TW3.append([])
        layer = layer.split(",")
        for value in layer:
            TW3[i].append(float(value))
        i += 1

    line = file.readline()
    line = line.replace("\n", "")
    line = line.split("/")
    line = line[:-1]
    i = 0
    for layer in line:
        Tb3.append([])
        layer = layer.split(",")
        for value in layer:
            Tb3[i].append(float(value))
        i += 1

TW1 = np.array(TW1)
Tb1 = np.array(Tb1)
TW2 = np.array(TW2)
Tb2 = np.array(Tb2)
TW3 = np.array(TW3)
Tb3 = np.array(Tb3)

print()
dev_predictions = make_predictions(X_test, TW1, Tb1, TW2, Tb2, TW3, Tb3)
print("Teacher Test Accuracy: ", get_accuracy(dev_predictions, Y_test))

# while True:
#     test = input("Input: ")
#     if test == "q":
#         break
#     test_prediction(int(test), TW1, Tb1, TW2, Tb2, TW3, Tb3)
#     print()
    
def init_params_student():
    W1 = np.random.rand(10, 784) - 0.5 
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def softmax_student(Z):
    return np.exp(Z / T) / sum(np.exp(Z / T))

def forward_prop_student(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax_student(Z2)
    return Z1, A1, Z2, A2

def back_prop_student(Z1, A1, Z2, A2, W2, X, Y):
    # one_hot_Y = one_hot(Y)
    # print(one_hot_Y)
    dZ2 = A2 - Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params_student(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def gradient_descent_student(X, Y, iterations, alpha, YT):
    W1, b1, W2, b2 = init_params_student()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop_student(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop_student(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params_student(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), YT))
    return W1, b1, W2, b2

def make_predictions_student(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop_student(W1, b1, W2, b2, X)
    return get_predictions(A2)

def test_prediction_student(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions_student(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
_, _, _, _, _, teacherY =  forward_prop(TW1, Tb1, TW2, Tb2, TW3, Tb3, X_train)
# print("TeacherY: ", teacherY)
SW1, Sb1, SW2, Sb2 = gradient_descent_student(X_train, teacherY, 500, 0.1, Y_train)

print()
dev_predictions = make_predictions_student(X_test, SW1, Sb1, SW2, Sb2)
print("Test Accuracy: ", get_accuracy(dev_predictions, Y_test))

while True:
    test = input("Input: ")
    if test == "q":
        break
    test_prediction_student(int(test), SW1, Sb1, SW2, Sb2)
    print()