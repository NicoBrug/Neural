#include <iostream>
#include <eigen3/Eigen/Dense>
#include "../header/network.h"
#include "../header/fc_layer.h"
#include "../header/activation_layer.h"
#include "../header/activation.h"
#include "../header/loss.h"
#include "../header/kernel.h"
#include <EigenRand/EigenRand>
#include <eigen3/Eigen/Core>

using namespace Eigen;
using Eigen::MatrixXd;
using namespace std;


int main() {

    //const auto processor_count = std::thread::hardware_concurrency();
    //cout << "core" << processor_count <<endl;

    std::vector<Eigen::Vector3d> v1(1000, Eigen::Vector3d{ 1.0, 10.0, 1111.0 });
    std::vector<Eigen::Vector3d> v2(1000, Eigen::Vector3d{ -1.0, 1.0, 1.0 });

    double x = Kernel::dot(v1,v2);
    cout << "GPU" << x << endl; 
    
    MatrixXd x_data(4,2);
    x_data << 
            9,2,
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

    ArrayXXd e = x_data.array();

    ArrayXXd* m1;

    cout << x_data << endl;

    cout << e << endl;

    cout << e(2) << endl;

    /* Network net;

    Loss* mse = new Mse();
    Loss* cre = new Cross_entropy();

    Activation* than = new Than();
    Activation* sigmoid = new Sigmoid();
    Activation* relu = new Relu();
    Activation* softplus = new SoftPlus();
    Activation* leakyrelu = new LeakyRelu(0.2);

    Layer* fcl1 = new Fc_Layer(2,5);
    Layer* acl1 = new Activation_layer(than);
    Layer* fcl2 = new Fc_Layer(5,1);
    Layer* acl2 = new Activation_layer(than);

    net.Use(mse);

    net.Add(fcl1);
    net.Add(acl1);
    net.Add(fcl2);
    net.Add(acl2);

    net.Fit(x_data,x_train,500,0.1);

    net.Predict(x_test); */
    cout << "nbthread" << Eigen::nbThreads( ) <<endl;
    return 0;
}