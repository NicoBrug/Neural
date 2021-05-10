#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include "../includes/network.h"
#include "../includes/fc_layer.h"
#include "../includes/activation_layer.h"
#include "../includes/flatten_layer.h"
#include "../includes/activation.h"
#include "../includes/loss.h"
#include "../includes/kernel.h"
#include "../includes/core.h"
#include "../includes/loader/mnist.h"
#include "../includes/conv_layer.h"

#include <EigenRand/EigenRand>
#include <eigen3/Eigen/Core>

using namespace Eigen;
using namespace std;



int main() {

    //Load MNIST
    mnist train("../dataset/MNIST/train-images-idx3-ubyte",
		     "../dataset/MNIST/train-labels-idx1-ubyte", 1000);

    mnist test("../dataset/MNIST/t10k-images-idx3-ubyte",
		     "../dataset/MNIST/t10k-labels-idx1-ubyte", 30);

    //Declare dimension for convolution layer & filter
    std::tuple<int, int, int> dimensions = std::make_tuple(28,28,1);
    std::tuple<int, int, int> filter = std::make_tuple(3,3,2);

    //Init network without pre-load configuration
    Network net; 

    //Create new activation & loss function
    Activation* than = new Than();
    Loss* mse = new Mse();

    //Tell to network to use mean squared error for compute loss
    net.Use(mse);

    //Create configuration of network
    Conv_layer* c1 = new Conv_layer(dimensions,filter,1,1);
    Activation_layer* acl1 = new Activation_layer(than);
    Flatten_layer* fl = new Flatten_layer();
    Fc_Layer* fc1 = new Fc_Layer(784*2,100);
    Activation_layer* acl2 = new Activation_layer(than);
    Fc_Layer* fc2 = new Fc_Layer(100,10);
    Activation_layer* acl3 = new Activation_layer(than);


    //Add the configuration to network
    net.Add(c1);
    net.Add(acl1);
    net.Add(fl);
    net.Add(fc1);
    net.Add(acl2);
    net.Add(fc2);
    net.Add(acl3);

    //Fit the network with MNIST image & label
    net.Fit(train.data.images,train.data.labels,35,0.1,1);

    //Make prediction for verify the network
    net.Predict(test.data.images);

    //Print true result
    cout << "result true \n" << test.data.labels << endl;
    
    return 0;
}