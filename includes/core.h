#pragma once
#ifndef CORE_H 
#define CORE_H

#include <iostream>
#include <string> 
#include <eigen3/Eigen/Dense>
#include <EigenRand/EigenRand>

class Core {
    public:
        Core() {};
        static Eigen::MatrixXd RandomMatrix(int rows, int cols, float min, float max);

    protected:

};

#endif