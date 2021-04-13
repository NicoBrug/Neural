#ifndef NETWORK_H 
#define NETWORK_H

#include <iostream>
#include <stdlib.h>
#include <eigen3/Eigen/Dense>
#include <vector> 
#include <string> 
#include <chrono>
#include <fstream>
#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>
#include <sciplot/sciplot.hpp>
#include "layer.h"
#include "fc_layer.h"
#include "activation_layer.h"

class Network
{
    public:
        Network();
        Network(std::string); //for load prexisting network
        ~Network();

        void Add(Layer *layer);
        void Fit(Eigen::MatrixXd x_train, Eigen::MatrixXd y_train, int epochs, double learning_rate);
        double Mse(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);
        //Likelihood function loss -> to implemented
        //Log loss (cross entrpy loss) -> to implemented
        Eigen::MatrixXd Mse_prime(Eigen::MatrixXd y_true, Eigen::MatrixXd y_pred);
        std::vector<Eigen::MatrixXd> Predict(Eigen::MatrixXd input_data);
        bool Save(std::string);
        void Load(std::string);

    private:
        std::vector<Layer*> m_layer;
        std::vector<double> m_error;
        std::ofstream file;
        void PlotData(int epoch, std::vector<double> error);

};

#endif