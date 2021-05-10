Exemple                         
============


## XOR 
This example consists in using a multilayer perceptron to solve the XOR problem.
To understand this example, let's look at this diagram for a moment. 

\image html images/XORproblem.jpeg width=400px

First of all, it is necessary to create the data we need to train the neural network. 
```
MatrixXd x_data(4,2);
x_data << 
            0,0,
            0,1,
            1,0,
            1,1;

MatrixXd x_valid(4,1);
x_train <<  0,
            1,
            1,
            0;
```
As we can see, the matrices correspond to the inputs and outputs of the problem to solve.

We need also to create the test MAtrix for validation result. 
```
MatrixXd x_test(4,2);
x_test << 
            1,1, // -> 0 
            0,1, // -> 1
            0,0, // -> 0
            1,0; // -> 1
```

The data configuration is finished, we can now create our neural network!

```
Network net;
```

Now that our network is declared, we can also create the different activation functions, and the loss function that we will use to solve this problem.

```
Loss* mse = new Mse(); 
Activation* than = new Than();
```

In order to build the structure of our network, we must create the different layers of it:

```
Fc_Layer* fcl1 = new Fc_Layer(2,5);
Activation_layer* acl1 = new Activation_layer(than);
Fc_Layer* fcl2 = new Fc_Layer(5,1);
Activation_layer* acl2 = new Activation_layer(than);
```

The first layer has two inputs (0 and 1 which represents a sample of the previously created data set), and 5 outputs. 
The last layer has 5 inputs that correspond to the output of layer 1, and 1 output that represents the value we want to predict. 

Now let's add the layers to the networks 

```
net.Add(fcl1);
net.Add(acl1);
net.Add(fcl2);
net.Add(acl2);
```

All we have to do is to train the network, specifying which loss function we want to use and check its operation.

```
net.Use(mse);
net.Fit(x_data,x_train,100,0.1,1);
```
Here, we pass to the "fit" function the input data set, the result set, the number of epochs, the learning rate, and the batch size. The dataset being minimal, we set the batch size to 1.

We want to test our network on the test data set created earlier. So we can write :

```
net.Predict(x_test);
```

Now you can compile and observe your results!

## MNIST Dense
We will now turn to a slightly more complicated problem: Solving the MNIST. The mnist database, is a set of data representing black and white numbers written by hand. It is a very well known dataset in the machine learning world because there has been a lot of research on it. Indeed, banks use this kind of algorithms to find automatically which numbers you have entered on your bank draft! It's better that the algorithm knows the difference between a 1 and a 9 you understand why ;) 
We will use the same neural structure in full connected as for the XOR example. By changing of course our input and the number of hidden layer. 

First, you need to import the MNIST data. Fortunately, Neural has a function that allows you to quickly import and manipulate the MNIST dataset. 

```
mnist train("../dataset/MNIST/train-images-idx3-ubyte",
		     "../dataset/MNIST/train-labels-idx1-ubyte", 54000);

mnist test("../dataset/MNIST/t10k-images-idx3-ubyte",
		     "../dataset/MNIST/t10k-labels-idx1-ubyte", 30);
```
We import 54000 samples for the training, and we want to check the network on 30 samples.
We must now create our neural network. This problem consumes more resources than the XOR problem, so we can use multiple threads like this: 
```
Network net;
net.SetThreads(3);
```
```
Activation* than = new Than();
Loss* mse = new Mse();
net.Use(mse);
```
A MNIST image is 28 px*28px (in black and white), so we have to assign 28*28 neurons for the first layer. The last layer represents the list of values that the network will predict. We want to predict a number between 0 and 9, so we have 10 neurons that correspond to the probability of the match with the input set.
```
Fc_Layer* fcl1 = new Fc_Layer(784,128);
Activation_layer* acl1 = new Activation_layer(than);
Fc_Layer* fcl2 = new Fc_Layer(128,10);
Activation_layer* acl2 = new Activation_layer(than);
net.Add(fcl1);
net.Add(acl1);
net.Add(fcl2);
net.Add(acl2);
```
We can now train our network: 

```
net.Fit(train.data.images,train.data.labels,4,0.1,1);
net.Predict(test.data.images);
cout << "result true \n" << test.data.labels << endl;
```

## MNIST Conv
Neural is a library developed in C++ allowing the realization of artificial neural networks. It is intended to be simple to use and easy to install (Just create the Docker and start to code!). If you encounter any problem, don't hesitate to let me know. 

It is built on the basis of eigen which allows to facilitate the manipulation of data structures, and to have various functions of manipulation of matrices or lists. Moreover, eigen has natively a multi-threading function which allows you to easily add the number of threads you want to allocate to optimize the performances you need.

