# Neural


## ℹ️️ Description

## 🔧 How to install the development environment
Ok, first let's start by building the image (the development environment). 
```
docker build -t neural .
```
Now we need to create the container. To do this we need to specify that we want to "share" the folder with our container. 
```
docker run -it -v ~/pathoffolder:/home/Projects/Neural -t neural /bin/bash
```
Exemple : if my Foler is in Document/Projects/Neural
```
docker run -it -v ~/Documents/Projects/Neural:/home/Projects/Neural -t neural /bin/bash
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

## 🔧 How to use

Create the network
```
Network net; 
```
or if we already save a Neural network :
```
Network net("../net.json"); 
```
Instantiate the different activation function :
```
Activation* than = new Than();
Activation* sigmoid = new Sigmoid();
```
Instantiate the different Layer : Fc_Layer = full connected neuron
```
Fc_Layer* fcl1 = new Fc_Layer(2,5);
Activation_layer* acl1 = new Activation_layer(than);
```
Add the different layer to Network :
```
Fc_Layer* fcl1 = new Fc_Layer(2,5);
Activation_layer* acl1 = new Activation_layer(than);
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

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact
Nicolas Brugie - nicolasbrugie@gmail.com


