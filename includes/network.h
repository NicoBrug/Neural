/**
 * \class Network
 *
 *
 * \brief Provide an example
 *
 * this class is the root class to create a neural network. 
 * Instantiate this class to add layers to the network, train the network, save the weights and load a pre-existing network.
 *
 * \note Attempts at zen rarely work.
 *
 * \author (last to touch it) $Author: Brugie Nicolas $
 *
 *
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
#include "layer.h"
#include "fc_layer.h"
#include "activation_layer.h"
#include "loss.h"

class Network
{
    public:
        Network();
        Network(std::string); //for load prexisting network
        ~Network();

        void Add(Layer *layer);
        void Fit(Eigen::MatrixXd x_train, Eigen::MatrixXd y_train, int epochs, double learning_rate, int batch_size);
        void Use(Loss *l);
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

#endif