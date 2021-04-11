import numpy as np


def tanh(x):
    print("X",x, "than", np.tanh(x))
    return np.tanh(x)

def tanh_prime(x):
    print("X_Prime",x, "than_prime", 1-np.tanh(x)**2)
    return 1-np.tanh(x)**2

# loss function and its derivative
def mse(y_true, y_pred):
    
    return np.mean((y_true-y_pred)**2)

def mse_prime(y_true, y_pred):

    return 2*(y_pred-y_true)/y_true.size