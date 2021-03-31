import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from utils import tanh, tanh_prime
from utils import mse, mse_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

V_train = np.array([[[0,0]]])

# network
net = Network()

net.add(FCLayer(2, 5))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(5, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(V_train)
print(out)