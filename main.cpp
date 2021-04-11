#include <iostream>
#include <eigen3/Eigen/Dense>
#include "header/network.h"
#include "header/fc_layer.h"
#include "header/activation_layer.h"

using namespace Eigen;
using Eigen::MatrixXd;
using namespace std;

// g++ main.cpp src/network.cpp src/fc_layer.cpp src/activation_layer.cpp -I/usr/local/include/ -o Programme
// sudo apt-get install libjsoncpp-dev


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


    cout << "Data \n" << x_data << endl;
    cout << "Train \n" << x_train << endl;
    
    //Create Network
    Network net;

    //Create Layer -> Fc Layer = full connected layer    
    Fc_Layer* fcl1 = new Fc_Layer(2,5);
    Activation_layer* acl1 = new Activation_layer();
    Fc_Layer* fcl2 = new Fc_Layer(5,1);
    Activation_layer* acl2 = new Activation_layer();

    //Add different layer to Network
    net.Add(fcl1);
    net.Add(acl1);
    net.Add(fcl2);
    net.Add(acl2);

    // Train the network 
    net.Fit(x_data,x_train,1000,0.1);
    
    //Call predict test
    net.Predict(x_test);
    
    //Save (in json)
    net.Save("My_Amazing_Weights");

    //Load the weights
    net.Load("../net.json");

    return 0;
}