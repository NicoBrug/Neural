#pragma once
#ifndef CORE_H 
#define CORE_H

#include <iostream>
#include <string> 
#include <random> 
#include <eigen3/Eigen/Dense>
#include <EigenRand/EigenRand>

class Core {
    public:
        Core() {};
        static Eigen::MatrixXd RandomMatrix(int rows, int cols, float min, float max);
        static Eigen::MatrixXd Correlate2D(Eigen::MatrixXd input, Eigen::MatrixXd filter, int stride, std::string padding);
        static Eigen::MatrixXd Padding(Eigen::MatrixXd m, int p);

        static void PrintArray(double *array);
    protected:

};

#endif