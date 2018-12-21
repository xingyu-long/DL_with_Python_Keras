import numpy as np


# keras.layers.Dense(512, activation = 'relu')
# output = relu(dot(W, input) + b)

def native_relu(x):
    # x is 2D Numpy tensor
    assert len(x.shape) == 2

    x = x.copy() # Avoid overwriting the input tensor
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


# test for relu
# x = np.array([[1, 2, 3],
#               [-4, -5, 6]
#               ])
# relu_x = native_relu(x)
# print(relu_x)


# element-wise addition
def native_add(x, y):
    # x and y are 2D Numpy tensor
    assert len(x.shape) == 2
    assert y.shape == x.shape

    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


# test for e-w addition
# x = np.array([[1, 2, 3],
#               [-4, -5, 6]
#               ])
# y = np.array([[2, 5, 6],
#               [-4, -5, 6]
#               ])
# z = native_add(x, y)
# print(z)


