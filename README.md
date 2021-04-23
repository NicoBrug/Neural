
![Screenshot](pics/logo.png?raw=true )
## Description
Neural is a library developed in C++ allowing the realization of artificial neural networks. It is intended to be simple to use and easy to install (Just create the Docker and start to code!). If you encounter any problem, don't hesitate to let me know. 

## Getting Started
To get a local copy up and running follow these simple steps.
### Prerequisites
* Docker
* nvidia-docker2
```
git clone https://github.com/NicoBrug/Neural.git
```
## How to install the development environment
Ok, first let's start by building the image (the development environment). 
```
xhost local:root
docker build -t neural-cuda .
```
Now we need to create the container. To do this we need to specify that we want to "share" the folder with our container. (path = The path to the folder where Neural is located )
```
sudo docker run \
    -it \
    --runtime=nvidia \
    --net=host \
    --privileged \
    -e DISPLAY=unix$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/Documents/neuralnetworks/Neural:/home/Projects/Neural \
    -t neural-cuda /bin/bash
```
Now we want to compile the project and check that the development environment is ready!
```
mkdir build
cd build
cmake ..
make
```
Everything to compile without problems? Then you can go to the next step, run the XOR example.
```
./Neural
```

## How to use

Create the network
```
Network net; 
```
or if we already save a Neural network :
```
Network net("../net.json"); 
```
Do you want to assign a number of threads to optimize the operation? 
```
net.SetThread(5); 
```
If you don't know how much core your pc has -> 
```
cout << "my cores: " << net.GetThreads << endl;
```
Instantiate the different activation function :
```
Activation* than = new Than();
Activation* sigmoid = new Sigmoid();
Activation* relu = new Relu();
Activation* softplus = new SoftPlus();
```
Instantiate the loss function & use it in Network
```
Loss* mse = new Mse();
Loss* cre = new Cross_entropy();
net.Use(mse);
```
Instantiate the different Layer : Fc_Layer = full connected neuron
```
Fc_Layer* fcl1 = new Fc_Layer(2,5);
Activation_layer* acl1 = new Activation_layer(than);
```
Add the different layer to Network :
```
net.Add(fcl1);
net.Add(acl1);
```
Train the network : 
```
net.Fit(x_data,x_train,1000,0.1);
```
Use the network :
```
net.Predict(x_test);
```
Save :
```
net.Save("My_Amazing_Weights");
```
Plot:
```
net.Plot("loss");
```

## Exemple
### MNIST
Use of the stochastic gradient descent method with the mean square error calculation function.
Loss 
![Screenshot](pics/mnistloss.png?raw=true )

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Nicolas Brugie - nicolasbrugie@gmail.com


