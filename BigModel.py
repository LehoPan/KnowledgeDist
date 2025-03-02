import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
Cumbersome model adapted from neural network by Samson Zhang on youtube:
https://www.youtube.com/watch?v=w8yWXqWQYmU
'''

data = pd.read_csv('train.csv') # Training MNIST Data from Kaggle

data = np.array(data)           # Change to numpy array
m, n = data.shape
np.random.shuffle(data)         # Randomize data order everytime we train
T = 10                          # Temperature for softer distribution of probabilities when distilling

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

# Increased Complexity for this cumbersome model
def init_params():
    # Here we have 3 hidden layers, each with 40 neurons instead of 10, the final hidden layer is 10 still
    W1 = np.random.rand(40, 784) - 0.5  # Weights for Layer 1
    b1 = np.random.rand(40, 1) - 0.5    # Biases for Layer 1

    W2 = np.random.rand(40, 40) - 0.5   # Weights Layer 2
    b2 = np.random.rand(40, 1) - 0.5    # Biases Layer 2

    W3 = np.random.rand(10, 40) - 0.5   # Weights Layer 3
    b3 = np.random.rand(10, 1) - 0.5    # Biases Layer 3
    return W1, b1, W2, b2, W3, b3

# Rectified Linear Unit as our Activation Function
def ReLU(Z):
    return np.maximum(0, Z)

# Softmax Function, with the addition of adding the temperature parameter for spreading out probabilities
def softmax(Z):
    return np.exp(Z / T) / sum(np.exp(Z / T))

# Forward Propagation algorithm
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)                # A3 is our output layer
    return Z1, A1, Z2, A2, Z3, A3

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

# Updates weights and biases after backprogagating, 
# also includes alpha (Learning Rate) hyperparameter in equation
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

# Converts the output layer's probabilities into one single numerical prediction as output
def get_predictions(A2):
    return np.argmax(A2, 0)

# Compares the predictions to the a list of labels in the same order, and returns the accuracy
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

# Training function to iterate over data set and train the model by tuning the weights and biases
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 50 == 0:             # Print the accuracy every 50 iterations
            print("Iteration ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A3), Y))
    return W1, b1, W2, b2, W3, b3

# This is for feeding input data to the model and getting predictions,
# without invoking back propagation and training the model. Purely for output.
def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    return get_predictions(A3)

# Retrieves an example in the training set at the desired index and feeds it to the model
# Renders an image of the example fed, and prints out the prediction as well as the intended label.
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
    
# Main driver code for the model
if __name__ == "__main__":

    # Trains the cumbersome model
    W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 500, 0.1)

    # Since this model takes significant time to train, we save all the values in each layer to a text file
    with open("teacher.txt", "w") as file:
        # Each weight and bias is on its own line
        # Each set of weights or biases for each neuron is separated by a "/"
        # Each value is separated by a ","
        for layer in W1:
            file.write(str(layer[0]))
            for value in layer[1:]:
                file.write(",")
                file.write(str(value))
            file.write("/")
        file.write("\n")
        
        for layer in b1:
            file.write(str(layer[0]))
            for value in layer[1:]:
                file.write(",")
                file.write(str(value))
            file.write("/")
        file.write("\n")

        for layer in W2:
            file.write(str(layer[0]))
            for value in layer[1:]:
                file.write(",")
                file.write(str(value))
            file.write("/")
        file.write("\n")

        for layer in b2:
            file.write(str(layer[0]))
            for value in layer[1:]:
                file.write(",")
                file.write(str(value))
            file.write("/")
        file.write("\n")

        for layer in W3:
            file.write(str(layer[0]))
            for value in layer[1:]:
                file.write(",")
                file.write(str(value))
            file.write("/")
        file.write("\n")

        for layer in b3:
            file.write(str(layer[0]))
            for value in layer[1:]:
                file.write(",")
                file.write(str(value))
            file.write("/")
        file.write("\n")

    print()
    dev_predictions = make_predictions(X_test, W1, b1, W2, b2, W3, b3)  # Tests accuracy on test data set
    print("Test Accuracy: ", get_accuracy(dev_predictions, Y_test))     # Prints accuracy to console

    # Loop for allowing the user to choose a test case to run through the model
    while True:
        test = input("Input: ")
        if test == "q":
            break
        test_prediction(int(test), W1, b1, W2, b2, W3, b3)
        print()
    
