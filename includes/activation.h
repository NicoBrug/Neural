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
            return  (1 / (1 + exp(-x.array()))) * (1 - (1 / (1 + exp(-x.array()))));
        }
};

class Softmax : public Activation {
    public:
        Softmax() {
            m_type= "Softmax";
        };
        virtual Eigen::MatrixXd Acti(Eigen::MatrixXd x){
            Eigen::MatrixXd expo = exp(x.array()-x.maxCoeff());
            return expo/expo.sum();
        }
        virtual Eigen::MatrixXd Acti_prime(Eigen::MatrixXd x){
            Eigen::MatrixXd exposum = (exp(x.array()-x.maxCoeff()))/(exp(x.array()-x.maxCoeff()).sum());
            return exposum.array()*(1-exposum.array()); 
        }
};

class Relu : public Activation {
    public:
        Relu() {
            m_type= "Relu";
        };
        virtual Eigen::MatrixXd Acti(Eigen::MatrixXd x){
            return x.cwiseMax(0.0);
        }
        virtual Eigen::MatrixXd Acti_prime(Eigen::MatrixXd x){
            for (int r(0); r<x.rows();r++){
                for (int c(0); c<x.cols();c++){
                    if (x(r,c)<0){
                        x(r,c) = 0;
                    }
                    else{
                        x(r,c) = 1;
                    }
                }
            }
            return  x;
        }
};

class SoftPlus : public Activation {
    public:
        SoftPlus() {
            m_type = "SoftPlus";
        };
        virtual Eigen::MatrixXd Acti(Eigen::MatrixXd x){
            return log(1+(exp(x.array()))); //y = log(1+exp(x)))
        }
        virtual Eigen::MatrixXd Acti_prime(Eigen::MatrixXd x){
            return  1/(1+exp(-x.array())); //dy/dx = 1/(1+exp(-x))
        }
};

class LeakyRelu : public Activation {
    public:
        LeakyRelu(double a) {
            m_type= "LeakyRelu";
            m_alpha = a;
        };

        double m_alpha;

        virtual Eigen::MatrixXd Acti(Eigen::MatrixXd x){
            return x.cwiseMax(0.0);
        }
        virtual Eigen::MatrixXd Acti_prime(Eigen::MatrixXd x){
            for (int r(0); r<x.rows();r++){
                for (int c(0); c<x.cols();c++){
                    if (x(r,c)<=0){
                        x(r,c) = m_alpha;
                    }
                    else{
                        x(r,c) = 1;
                    }
                }
            }
            return  x;
        }
};
#endif