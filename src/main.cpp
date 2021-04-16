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

void print(double *array){
    for (int i = 0; i <= sizeof(array); ++i) {
        cout << array[i] << "  ";
    }
    cout << "\n" << endl;
}


int main() {

    //const auto processor_count = std::thread::hardware_concurrency();
    //cout << "core" << processor_count <<endl;

    std::vector<Eigen::Vector3d> v1(1000, Eigen::Vector3d{ 1.0, 10.0, 11.0 });
    std::vector<Eigen::Vector3d> v2(1000, Eigen::Vector3d{ -1.0, 1.0, 1.0 });
    double x = Kernel::dot(v1,v2);
    cout << "GPU " << x << endl; 

    Matrix<double, 3, 3, RowMajor> Ap;
    Ap << 1,1,8, 
          0,1,9, 
          0,0,15;

    Matrix<double, 3, 3, RowMajor> Ac;
    Ac << 1,1,1, 
          32.5,1,98, 
          0,5,1;

    Eigen::ArrayXXd result;
    double* res  = new double[9];
    cout << Ap.rows() << Ac.cols() << endl;
    double* res1 = Kernel::dot(Ap.data(), Ac.data(),3); 

    cout << "GPU : \n" << endl;
    cout << Map<Matrix<double,3,3,RowMajor> >(res1) << endl;

    cout << "\n" << endl ;

    cout << "CPU : \n"  << Ap * Ac << endl ;

    cout << "nbthread" << Eigen::nbThreads( ) <<endl;
    return 0;
}