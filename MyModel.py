import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
This is the initial neural network from Samson Zhang on youtube:
https://www.youtube.com/watch?v=w8yWXqWQYmU
'''

data = pd.read_csv('train.csv') # Training MNIST Data from Kaggle

data = np.array(data)           # Change to numpy array
m, n = data.shape
np.random.shuffle(data)         # Randomize data order everytime we train

# Partitions aside the first 1000 examples as the testing data set
data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.      # Divide by the max value of 255 so every value is from 0 and 1

# Second remaining partition is the training data examples
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.    # Again normalize data to a range of 0 and 1
_, m_train = X_train.shape

# Declares the Weights and Biases that make up our layers
def init_params():
    # Here we have two hidden layers, initilized at random values between -0.5 and 0.5 using numpy
    W1 = np.random.rand(10, 784) - 0.5  # Weights in Layer 1 in a Numpy Array
    b1 = np.random.rand(10, 1) - 0.5    # Biases of Layer 1 in np array

    W2 = np.random.rand(10, 10) - 0.5   # Weights of Layer 2 in np array
    b2 = np.random.rand(10, 1) - 0.5    # Biases of Layer 2 in np array
    return W1, b1, W2, b2

# Rectified Linear Unit as our Activation Function
def ReLU(Z):
    return np.maximum(0, Z)

# Softmax Function
def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

# Forward Propagation algorithm
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)        # A2 is our output layer, containing the probabilities for predictions
    return Z1, A1, Z2, A2

# Function to one hot encode all the numerical labels in the data set
# Convert into an np array filled with zeros, and a one at the index indicating the correct number
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Define the derivative of ReLU for back propagation
def deriv_ReLU(Z):
    return Z > 0

# Back propagation to tune weights and biases during training
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Updates weights and biases after backprogagating, 
# also includes alpha (Learning Rate) hyperparameter in equation
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

# Converts the output layer's probabilities into one single numerical prediction as output
def get_predictions(A2):
    return np.argmax(A2, 0)

# Compares the predictions to the a list of labels in the same order, and returns the accuracy
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# Training function to iterate over data set and train the model by tuning the weights and biases
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0: # Every 50 iterations print an output into the console on the current accuracy
            print("Iteration ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

# This is for feeding input data to the model and getting predictions,
# without invoking back propagation and training the model. Purely for output.
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    return get_predictions(A2)

# Retrieves an example in the training set at the desired index and feeds it to the model
# Renders an image of the example fed, and prints out the prediction as well as the intended label.
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Main driver code for the model
if __name__ == "__main__":
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1) # Trains the model

    print()
    dev_predictions = make_predictions(X_test, W1, b1, W2, b2)      # Tests accuracy on test data set
    print("Test Accuracy: ", get_accuracy(dev_predictions, Y_test)) # Prints accuracy to console

    # Loop for allowing the user to choose a test case to run through the model
    while True:
        test = input("Input: ")
        if test == "q":
            break
        test_prediction(int(test), W1, b1, W2, b2)
        print()
    
