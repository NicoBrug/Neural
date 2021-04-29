#pragma once
#ifndef LOSS_H 
#define LOSS_H

#include <iostream>
#include <string> 
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Jacobi>

namespace Neural
{
    class Loss {
        public:
            Loss() {};
            virtual double Compute(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred) = 0;
            virtual Eigen::MatrixXd Compute_prime(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred) = 0;

    };


    class Mse : public Loss {
        public:
            Mse() {};
            virtual double Compute(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred){
                Eigen::MatrixXd diff = y_true-y_pred;
                return diff.array().pow(2).mean();
            }
            virtual Eigen::MatrixXd  Compute_prime(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred){
                Eigen::MatrixXd diff = y_pred-y_true;  
                return (2*diff)/y_true.size();
            }
    };


    class Cross_entropy : public Loss {
        public:
            Cross_entropy() {};
            virtual double Compute(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred){
                double res;
                for (int i(0);i<y_true.cols();i++){
                    res+= y_true(0,i)*log(y_pred(0,i));
                }
                return -res;
            }
            virtual Eigen::MatrixXd  Compute_prime(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred){
                //Eigen::MatrixXd res = (-(y_true/y_pred)+((1-y_true)/(1-y_pred)))/y_true.size();
                return y_true;
            }
    };
}
#endif