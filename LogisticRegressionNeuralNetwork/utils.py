import numpy as np
import h5py

def load_dataset():
    """Function that performs loading of the dataset files and returns the training
    and testing sets and its classes
    """

    train_dataset = h5py.File("datasets/train_catvnoncat.h5", mode= "r")        # loads train dataset file from disk with readonly mode
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])                # takes train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])                # takes train set labels

    test_dataset = h5py.File("datasets/test_catvnoncat.h5", mode = "r")         # loads test dataset file from disk with readonly mode
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])                   # takes test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])                   # takes test set labels

    classes = np.array(test_dataset["list_classes"][:])                         # takes list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) # reshapes train set features to (1, m)
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))    # reshapes test set features to (1, m)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def sigmoid(z):
    """Function that performs the calculation of the sigmoid of z
    """

    s = 1 / (1 + np.exp(-z))                                                 # computes sigmoid of z

    return s

def initialize_with_zeros(dim):
    """Function that initializes the weight vector and the bias
    """

    w = np.zeros((dim, 1))                                                      # initializes w vector with shape (dim , 1)
    b = 0                                                                       # initializes bias with 0

    return w, b


def propagate(w, b, X, Y):
    """Function that performs forward and backward propagation steps for the
    learning parameters
    """
    m = X.shape[1]

    # forward propagation (from X to cost)
    A = sigmoid(np.dot(w.T, X) + b)                                             # computes activation
    cost = -(1 / m) * np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A))         # computes cost

    # backward propagation (to find grad)
    dw = (1 / m) * X.dot((A - Y).T)                                             # computes derivative of w
    db = (1 / m) * np.sum(A - Y)                                                # computes derivative of b

    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """Function that optimizes w and b by running a gradient descent algorithm
    """

    costs = []

    for i in range(num_iterations):        
        grads, cost = propagate(w, b, X, Y)                                     # cost and gradient calculation

        dw = grads["dw"]                                                        # retrieves derivative of w from grads
        db = grads["db"]                                                        # retrieves derivative of b from grads

        w = w - (learning_rate * dw)                                            # updates weights
        b = b - (learning_rate * db)                                            # updates bias

        if i % 100 == 0:
            costs.append(cost)                                                  # records the costs
        
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))                     # prints the cost for every 100 training examples

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w, b, X):
    """Function that predicts whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    """

    m = X.shape[1]                                                              # gets the number of examples
    Y_prediction = np.zeros((1, m), dtype = int)                                # initializes vector of probabilities
    w = w.reshape(X.shape[0], 1)                                                # reshapes weights vector with shape (nx, 1)

    # compute vector A predicting the probabilities
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0                                # converts probabilities A[0, i] to actual predictions p[0, i]

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """Function that builds the logistic regression model
    """

    w, b = initialize_with_zeros(X_train.shape[0])                              # initializes parameters with zeros

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)  # gradient descent

    w = parameters["w"]                                                         # retrieves parameter w
    b = parameters["b"]                                                         # retrieves parameter b

    Y_prediction_test = predict(w, b, X_test)                                   # predicts test set examples
    Y_prediction_train = predict(w, b, X_train)                                 # predicts train set examples

    print("Train set accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100)) # prints train errors
    print("Test set accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))    # prints test errors

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d