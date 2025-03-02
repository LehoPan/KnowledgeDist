import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
T = 10                   # Temperature for softening probability distributions

# Partition data set into test set
data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

# Partition data set into training set
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

# Rectified Linear Unit
def ReLU(Z):
    return np.maximum(0, Z)

# Soft max with Temperature Hyper Parameter
def softmax(Z):
    return np.exp(Z / T) / sum(np.exp(Z / T))

# Forward Propogation for 3 hidden layers
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Derivative of ReLU for back propogation
def deriv_ReLU(Z):
    return Z > 0

# Changing probabilities to a single number for output
def get_predictions(A2):
    return np.argmax(A2, 0)

# Returns an accuracy against the given labels
def get_accuracy(predictions, Y):
    # print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# Isolates the final output layer for the teacher model
def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    return get_predictions(A3)
    

# Initializes the student's layers
def init_params_student():
    # 2 hidden layers, both with 10 neurons each
    W1 = np.random.rand(10, 784) - 0.5 
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

# Forward propogation
def forward_prop_student(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)        # A2 is our output layer
    return Z1, A1, Z2, A2

# Back propogation is modified. Y isn't the labels from the data set anymore
# Y is instead the probabilities from the teacher model
def back_prop_student(Z1, A1, Z2, A2, W2, X, Y):
    dZ2 = A2 - Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Update parameters of the student model with respect to alpha (learning rate) hyperparameter
def update_params_student(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Gradient descent for training the student model with forward/back propogation
def gradient_descent_student(X, Y, iterations, alpha, YT):
    W1, b1, W2, b2 = init_params_student()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop_student(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop_student(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params_student(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:             # Print an accuracy update to the console every 50 iterations
            print("Iteration ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), YT))
    return W1, b1, W2, b2

# Isolates the output layer for testing
def make_predictions_student(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop_student(W1, b1, W2, b2, X)
    return get_predictions(A2)

# Inserts and renders a test input into the student model and prints out the prediciton and label
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

# Driver code for the knowledge distillation
if __name__ == "__main__":
    # Initializes all the teacher values to be read from the pretrained file values
    TW1, Tb1, TW2, Tb2, TW3, Tb3 = [], [], [], [], [], []

    # Parses the teacher values file to have the same pretrained cumbersome model
    with open("teacher.txt", "r") as file:
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

    # Converts all the arrays to numpy arrays
    TW1 = np.array(TW1)
    Tb1 = np.array(Tb1)
    TW2 = np.array(TW2)
    Tb2 = np.array(Tb2)
    TW3 = np.array(TW3)
    Tb3 = np.array(Tb3)

    print()
    # Does a quick output to the console for the teacher accuracy against the testing data
    dev_predictions = make_predictions(X_test, TW1, Tb1, TW2, Tb2, TW3, Tb3)
    print("Teacher Test Accuracy: ", get_accuracy(dev_predictions, Y_test))

    # First gets all the teacher's predictions on the test data set, before they get processed into a prediction
    # In other words the output layer's set of probabilities
    _, _, _, _, _, teacherY =  forward_prop(TW1, Tb1, TW2, Tb2, TW3, Tb3, X_train)

    # Then pipes that into the training for the student, so the student isn't training with the test data labels
    # But instead what the teacher predicted instead
    SW1, Sb1, SW2, Sb2 = gradient_descent_student(X_train, teacherY, 500, 0.1, Y_train)

    print()
    # Prints to the console the student's accuracy against the testing data set
    dev_predictions = make_predictions_student(X_test, SW1, Sb1, SW2, Sb2)
    print("Test Accuracy: ", get_accuracy(dev_predictions, Y_test))

    # Loop to allow user to input and render sample training cases one at a time to the student model
    while True:
        test = input("Input: ")
        if test == "q":
            break
        test_prediction_student(int(test), SW1, Sb1, SW2, Sb2)
        print()