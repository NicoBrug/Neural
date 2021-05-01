/**
 * @file network.h
 * @brief This class allows to create the root structure of the network. 
 * @author Brugie Nicolas <nicolasbrugie@gmail.com>
 */



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
#include <omp.h>
#include <thread>
#include "layers/layer.h"
#include "layers/fc_layer.h"
#include "layers/activation_layer.h"
#include "loss.h"

namespace Neural
{
    class Network
    {
        public:
            Network();
            Network(std::string); //for load prexisting network
            ~Network();

            void Add(Layer *layer);
            void Fit(Eigen::MatrixXd x_train, Eigen::MatrixXd y_train, int epochs, double learning_rate, int batch_size);
            void Use(Loss *l);
            void Evaluate(Eigen::MatrixXd y_tests, Eigen::MatrixXd y_true);

            std::vector<Eigen::MatrixXd> Predict(Eigen::MatrixXd input_data);
            bool Save(std::string);
            void Load(std::string);
            void SetThreads(int thread);
            int GetThreads();


        private:
            Loss* m_loss; 
            std::vector<Layer*> m_layer;
            std::vector<double> m_error;
            std::ofstream file;
            void PlotData(int epoch, std::vector<double> error);

    };
}
#endif