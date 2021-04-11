#include "../header/network.h"
#include "../header/fc_layer.h"

#include <jsoncpp/json/json.h>
#include <jsoncpp/json/writer.h>

using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;

Network::Network(){
};

Network::~Network(){
    for (int i(0); i< m_layer.size(); i++){
        delete(m_layer[i]);
    }
};

void Network::Add(Layer *layer){
    m_layer.push_back(layer);
};

vector<MatrixXd> Network::Predict(MatrixXd input_data){
    int samples = input_data.rows();

    vector<MatrixXd> res;

    for (int i(0); i< samples; i++){
        MatrixXd output = input_data.row(i);

        for(int l(0); l < m_layer.size(); l++){
            output = m_layer[l]->Forward_propagation(output);
        }

        res.push_back(output);
        cout << "predict \n" << output << endl; 

    };

    return res;
};

void Network::Fit(MatrixXd x_train, MatrixXd y_train, int epochs, double learning_rate){
    int samples = x_train.rows();
    int cols = x_train.cols();

    cout << "samples " << samples << " cols " << cols << endl;

    for (int i(0);i<epochs; i++){
        double err(0);

        for (int j(0); j<samples; j++){
            
            MatrixXd output(1,cols);
            output = x_train.row(j);
            

            for (int l(0);l<m_layer.size();l++){
                output = m_layer[l]->Forward_propagation(output);
            } 

            err += this->Mse(y_train.row(j),output);
            MatrixXd error = this->Mse_prime(y_train.row(j),output);

            for (int k(m_layer.size()-1); k>=0; k--){
                error = m_layer[k]->Backward_propagation(error,learning_rate);
            }   
        }

        cout << "epoch " << i+1 << " | " << "error " << err/samples << endl;
        m_error.push_back(err/samples); // for plotting
    }
}


double Network::Mse(MatrixXd y_true, MatrixXd y_pred){
    MatrixXd diff = y_true-y_pred;
    return diff.array().pow(2).mean();
}


MatrixXd Network::Mse_prime(MatrixXd y_true, MatrixXd y_pred){
    MatrixXd diff = y_pred-y_true;  
    return 2*diff/y_true.size();
}

bool Network::Save(string name){
    file.open (name+".json");

    Json::Value event;   

    for (int l(0); l< m_layer.size();l++){
        if (m_layer[l]->AsWeights()){
            MatrixXd weights = static_cast<Fc_Layer*>(m_layer[l])->GetWeights();

            for (int i(0); i<weights.rows();i++){
                Json::Value vec(Json::arrayValue);

                for (int j(0); j<weights.cols();j++){
                    vec.append(Json::Value(weights(i,j)));
                }

                event["Network"]["Layer_"+to_string(l)][i] = vec;
            }

        }

    } 

    //std::cout << event << std::endl;

    file << event;

    file.close();

    return true;
}   


void Network::Load(string path){
     Json::Value root;
     std::ifstream file(path);
     file >> root;

     
     //cout << "root" << root;
}