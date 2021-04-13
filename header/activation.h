#pragma once
#ifndef ACTIVATION_H 
#define ACTIVATION_H

#include <iostream>
#include <string> 
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Jacobi>

class Activation {
    public:
        Activation() {};
        virtual Eigen::MatrixXd Acti(Eigen::MatrixXd x) = 0;
        virtual Eigen::MatrixXd Acti_prime(Eigen::MatrixXd x) = 0 ;
        std::string getType(){
            return this->m_type;
        };
    protected:
        std::string m_type;
};

class Than : public Activation {
    public:
        Than() {
            m_type= "Than";
        };
        virtual Eigen::MatrixXd Acti(Eigen::MatrixXd x){
            return x.array().tanh();
        }
        virtual Eigen::MatrixXd Acti_prime(Eigen::MatrixXd x){
            return  1-x.array().tanh().pow(2);
        }
};

class Sigmoid : public Activation {
    public:
        Sigmoid() {
            m_type= "Sigmoid";
        };
        virtual Eigen::MatrixXd Acti(Eigen::MatrixXd x){
            return 1/(1+exp(-x.array()));
        }
        virtual Eigen::MatrixXd Acti_prime(Eigen::MatrixXd x){
            return  1 / (1 + exp(-x.array())) * (1 - (1 / (1 + exp(-x.array()))));
        }
};

class Softmax : public Activation {
    public:
        Softmax() {
            m_type= "Softmax";
        };
        virtual Eigen::MatrixXd Acti(Eigen::MatrixXd x){
            Eigen::MatrixXd expo = exp(x.array());
            return expo/expo.sum();
        }
        virtual Eigen::MatrixXd Acti_prime(Eigen::MatrixXd x){
            return  x;
        }
};

#endif