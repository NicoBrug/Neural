#include <iostream>
#include <eigen3/Eigen/Dense>
#include <chrono>
#include "../includes/network.h"
#include "../includes/fc_layer.h"
#include "../includes/activation_layer.h"
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

    Matrix<double, 4, 2, RowMajor> Ap;
    Ap << 1,1, 
          0,1, 
          0,8,
          15,12;

    Matrix<double, 2, 3, RowMajor> Ac;
    Ac << 1,1,1, 
          32.5,1,98;
    
    MatrixXd m = Core::RandomMatrix(4,2,0,1);

    MatrixXd b = Ac;
    //cout << "Matrix : " << m << " \n" <<  endl ;
    //cout << "Matrix raw: " << m.data() << " \n" <<  endl ;

    cout << "GPU : \n" << endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    double* res1 = Kernel::dot(m.data(), b.data(),4,3,6); 
    auto t_end = std::chrono::high_resolution_clock::now();
    
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
    //cout << Map<Matrix<double,4,3,RowMajor> >(res1) << endl;
    cout << "time : " << elapsed_time_ms << " \n" <<  endl ;

    auto t_start1 = std::chrono::high_resolution_clock::now();
    MatrixXd m1 = m * b ;
    auto t_end1 = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms1 = std::chrono::duration<double, std::milli>(t_end1-t_start1).count();
    cout << "time : " << elapsed_time_ms1 << " \n" <<  endl ;
    
    cout << "nbthread" << Eigen::nbThreads( ) <<endl;


    /*---------------------------------------------------------------------------------------------*/
   
    //cout << "sample" << mnist_matrix.row(0) << " \nres" <<  mnist_label.row(0) <<endl;
    return 0;
}