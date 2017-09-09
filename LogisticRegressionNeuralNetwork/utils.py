import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', mode= 'r')        # load train dataset file from disk with readonly mode
    train_set_x_orig = np.array(train_dataset['train_set_x'][:])                # take train set features
    train_set_y_orig = np.array(train_dataset['train_set_y'][:])                # take train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', mode = 'r')         # load test dataset file from disk with readonly mode
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])                   # take test set features
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])                   # take test set labels

    classes = np.array(test_dataset['list_classes'][:])                         # take list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0])) # reshape train set features to (1, m)
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))    # reshape test set features to (1, m)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes