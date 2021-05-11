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


    Loss* mse = new Mse(); 
    Loss* cre = new Cross_entropy(); 
    net.Use(mse);
    
    Activation* than = new Than();

    Fc_Layer* fcl1 = new Fc_Layer(2,5);
    Activation_Layer* acl1 = new Activation_Layer(than);
    Fc_Layer* fcl2 = new Fc_Layer(5,1);
    Activation_Layer* acl2 = new Activation_Layer(than);

    net.Add(fcl1);
    net.Add(acl1);
    net.Add(fcl2);
    net.Add(acl2);

    net.Fit(x_data,x_train,100,0.1,1);

    net.Predict(x_test);

    return 0;

}