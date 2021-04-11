#ifndef NETWORK_H 
#define NETWORK_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <vector> 
#include <string> 
#include "layer.h"
#include <fstream>

class Network
{
    public:
        Network();
        ~Network();

        void Add(Layer *layer);
        void Fit(Eigen::MatrixXd x_train, Eigen::MatrixXd y_train, int epochs, double learning_rate);
        double Mse(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);
        Eigen::MatrixXd Mse_prime(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);
        std::vector<Eigen::MatrixXd> Predict(Eigen::MatrixXd input_data);
        bool Save(std::string);
        void Load(std::string);


    private:
        std::vector<Layer*> m_layer;
        std::vector<double> m_error;
        std::ofstream file;
};

#endif