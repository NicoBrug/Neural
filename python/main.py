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

z_test = np.array([[1],[2],[3],[4]])

x_true = np.array([[1],[2],[3],[4]])
x_pred = np.array([[1],[2],[6],[3]])

print("mseee", mse_prime(x_true,x_pred))
# network
net = Network()

net.add(FCLayer(2, 5))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(5, 1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
#net.fit(x_train, y_train, epochs=1, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)