#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include "../includes/network.h"
#include "../includes/layers/fc_layer.h"
#include "../includes/layers/activation_layer.h"
#include "../includes/layers/flatten_layer.h"
#include "../includes/layers/conv_layer.h"
#include "../includes/activation.h"
#include "../includes/loss.h"
#include "../includes/kernel.h"
#include "../includes/core.h"
#include "../includes/loader/mnist.h"

#include <EigenRand/EigenRand>
#include <eigen3/Eigen/Core>

using namespace Eigen;
using Eigen::MatrixXd;
using namespace std;
using namespace Neural;



int main() {

    MatrixXd x_data(4,2);
    x_data << 
            0,0,
            0,1,
            1,0,
            1,1;

    MatrixXd x_train(4,1);
    x_train <<  0,
                1,
                1,
                0;

    MatrixXd x_test(4,2);
    x_test << 
            1,1, // -> 0 //Result we waiting
            0,1, // -> 1
            0,0, // -> 0
            1,0; // -> 1


    Network net;

    Loss* mse = new Mse(); 
    Loss* cre = new Cross_entropy(); 
    net.Use(mse);
    
    Activation* than = new Than();

    Fc_Layer* fcl1 = new Fc_Layer(2,5);
    Activation_layer* acl1 = new Activation_layer(than);
    Fc_Layer* fcl2 = new Fc_Layer(5,1);
    Activation_layer* acl2 = new Activation_layer(than);

    net.Add(fcl1);
    net.Add(acl1);
    net.Add(fcl2);
    net.Add(acl2);

    net.Fit(x_data,x_train,100,0.1,1);

    net.Predict(x_test);

    

    return 0;
}