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

    int rows  = train.rows();
    int cols  = train.cols();
    
    int rowstest = test.rows();
    int colstest = test.cols();
    
    //Prepare data : train -> array of pixel matrix
    //Prepare data : label -> array[10] for output result 
    MatrixXd mnist_matrix(train.size(),rows*cols);
    MatrixXd mnist_label(train.size(),10);

    MatrixXd mnist_test(test.size(),rows*cols);
    MatrixXd mnist_test_label(test.size(),10);

    for (int i(0); i < train.size(); i++){
        std::vector<double> image = train.images(i);
        double size = cols * rows;
        mnist_matrix.row(i) = Map<Matrix<double,1,784> >(image.data());

        MatrixXd label(1,10);

        for (int j(0); j<10; j++){
            if (train.labels(i) == j){
                label(0,j) = 1.0;
            }
            else{
                label(0,j) = 0.0;
            }
        }
        mnist_label.row(i) = label;
    }

    for (int i(0); i < test.size(); i++){
        std::vector<double> image = test.images(i);
        double size = cols * rows;
        mnist_test.row(i) = Map<Matrix<double,1,784> >(image.data());

        MatrixXd label(1,10);

        for (int j(0); j<10; j++){
            if (test.labels(i) == j){
                label(0,j) = 1.0;
            }
            else{
                label(0,j) = 0.0;
            }
        }
        mnist_test_label.row(i) = label;
    }

    Network net;
    net.SetThreads(16);

    Activation* than = new Than();
    Activation* relu = new Relu();
    Activation* softmax = new Softmax();
    Activation* sigmoid = new Sigmoid();

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
    net.Fit(mnist_matrix,mnist_label,4,0.1,50);

    //net.Predict(mnist_test);

    //cout << "result true \n" << mnist_test_label << endl;
    

    return 0;

}