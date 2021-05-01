#include <iostream>
#include <eigen3/Eigen/Dense>
#include "../header/network.h"
#include "../header/fc_layer.h"
#include "../header/activation_layer.h"
#include "../header/activation.h"

using namespace Eigen;
using Eigen::MatrixXd;
using namespace std;

int main() {
    
    mnist train("../dataset/MNIST/train-images-idx3-ubyte",
		     "../dataset/MNIST/train-labels-idx1-ubyte", 54000);

    mnist test("../dataset/MNIST/t10k-images-idx3-ubyte",
		     "../dataset/MNIST/t10k-labels-idx1-ubyte", 30);

    
    Network net;
    net.SetThreads(16);

    Activation* than = new Than();

    Loss* mse = new Mse();
    net.Use(mse);

    Fc_Layer* fcl1 = new Fc_Layer(784,128);
    Activation_layer* acl1 = new Activation_layer(than);
    Fc_Layer* fcl2 = new Fc_Layer(128,10);
    Activation_layer* acl2 = new Activation_layer(than);


    net.Add(fcl1);
    net.Add(acl1);
    net.Add(fcl2);
    net.Add(acl2);


    
    //Fit network
    net.Fit(train.data.images,train.data.labels,4,0.1,50);

    //net.Predict(test.data.images);

    //cout << "result true \n" << test.data.labels << endl;
    

    return 0;

}