import utils
import time
import matplotlib.pyplot as plt

print("Started loading datasets")
tic = time.process_time()

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = utils.load_dataset()  # load the datasets

toc = time.process_time()
print("Ended loading datasets. Time elapsed: " + str(1000 * (toc - tic)) + "ms")

num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T             # reshape training set to (nx, m)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T                # reshape testing set to (nx, m)

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

print("Started modeling")
tic = time.process_time()

d = utils.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

toc = time.process_time()
print("Ended modeling. Time elapsed: " + str(1000 * (toc - tic)) + "ms")

index = 19
print("y = " + str(test_set_y[0, index]) + ", the prediction is that it is a \"" + classes[d["Y_prediction_test"][0, index]].decode("utf-8") +  "\" picture.")
plt.figure()
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
plt.show()