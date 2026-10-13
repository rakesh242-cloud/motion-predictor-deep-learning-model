import numpy as np

# load data
x_test = np.load('X_test.npy')
x_train = np.load('X_train.npy')
y_test = np.load('y_test.npy')
y_train = np.load('y_train.npy')

print("X_test shape",x_test.shape)
print("y_test shape",y_test.shape)
print("X_train shape",x_train.shape)
print("y_train shape",y_train.shape)

# # combine
# X = np.concatenate((x_train, x_test), axis=0)
# Y = np.concatenate((y_train, y_test), axis=0)

# print("Combined shape:", X.shape, Y.shape)

# # shuffle (important part)
# indices = np.random.permutation(len(X))

# X = X[indices]
# Y = Y[indices]

# # original sizes
# train_size = len(x_train)
# test_size = len(x_test)

# # split again
# x_train_new = X[:train_size]
# y_train_new = Y[:train_size]

# x_test_new = X[train_size:train_size + test_size]
# y_test_new = Y[train_size:train_size + test_size]

# # verify
# print("New shapes:")
# print(x_train_new.shape, y_train_new.shape)
# print(x_test_new.shape, y_test_new.shape)

# np.save('X_train_new.npy', x_train_new)
# np.save('y_train_new.npy', y_train_new)
# np.save('X_test_new.npy', x_test_new)
# np.save('y_test_new.npy', y_test_new)