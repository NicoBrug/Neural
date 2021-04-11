#ifndef NETWORK_H 
#define NETWORK_H

#include <iostream>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <vector> 
#include <string> 
#include "layer.h"
#include <fstream>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include "fc_layer.h"
#include "activation_layer.h"

class Network
{
    public:
        Network();
        Network(std::string);

        ~Network();

        void Add(Layer *layer);
        void Fit(Eigen::MatrixXd x_train, Eigen::MatrixXd y_train, int epochs, double learning_rate);
        double Mse(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);
        Eigen::MatrixXd Mse_prime(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);
        std::vector<Eigen::MatrixXd> Predict(Eigen::MatrixXd input_data);
        bool Save(std::string);
        void Load(std::string);


    private:
        Json::Value Serialise(Eigen::MatrixXd x, Eigen::MatrixXd b);

        std::vector<Layer*> m_layer;
        std::vector<double> m_error;
        std::ofstream file;
};

#endif