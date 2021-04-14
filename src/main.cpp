#include <iostream>
#include <eigen3/Eigen/Dense>
#include "../header/network.h"
#include "../header/fc_layer.h"
#include "../header/activation_layer.h"
#include "../header/activation.h"
#include "../header/loss.h"
#include <EigenRand/EigenRand>

using namespace Eigen;
using Eigen::MatrixXd;
using namespace std;


int main() {

    MatrixXd x_data(4,2);
    x_data << 
            0,0,
            0,1,
            1,0,
            1,1;

    MatrixXd x_train(4,1);
    x_train <<  0,
                0,
                1,
                1;

    MatrixXd x_test(4,2);
    x_test << 
            1,1, // -> 1 //Result we waiting
            0,1, // -> 0
            0,0, // -> 0
            1,0; // -> 1

    
    Network net;

    Loss* mse = new Mse();
    Loss* cre = new Cross_entropy();

    Activation* than = new Than();
    Activation* sigmoid = new Sigmoid();
    Activation* relu = new Relu();
    
    Layer* fcl1 = new Fc_Layer(2,5);
    Layer* acl1 = new Activation_layer(sigmoid);
    Layer* fcl2 = new Fc_Layer(5,1);
    Layer* acl2 = new Activation_layer(sigmoid);

    net.Use(mse);

    net.Add(fcl1);
    net.Add(acl1);
    net.Add(fcl2);
    net.Add(acl2);

    net.Fit(x_data,x_train,500,0.1);

    net.Predict(x_test);
    
    return 0;
}