#include "../header/network.h"
#include "../header/fc_layer.h"


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
    Json::Value vect(Json::arrayValue);

    for (int l(0); l< m_layer.size();l++){
        if (m_layer[l]->AsWeights()){
            MatrixXd weights = static_cast<Fc_Layer*>(m_layer[l])->GetWeights();
            MatrixXd bias = static_cast<Fc_Layer*>(m_layer[l])->GetBias();

            Json::Value json = ParseMatrix(weights,bias);
            vect.append(json);
        }

    }
    event["Network"] = vect;

    file << event;

    file.close();

    return true;
}   


void Network::Load(string path){

    Json::Value root;
    std::ifstream file(path);
    file >> root;

    const Json::Value net = root["Network"];
     for ( int i(0); i < net.size(); ++i ){
        Json::Value Layer = net[i];

        Json::Value weights = Layer["weights"];
        Json::Value bias = Layer["bias"];

        int input_size = weights.size();
        int output_size = weights[0].size();

        MatrixXd weights_matrix(input_size,output_size);
        MatrixXd bias_matrix(1,output_size);

        for ( int j(0); j < weights.size(); ++j ){
            Json::Value w = weights[j];

            for ( int k(0); k < w.size(); ++k ){
                double val = stod(w[k].asString()); 
                weights_matrix(j,k) = val;   
            }
        }
        
        Json::Value b = bias[0];
        cout << b.size() << endl;
        for ( int j(0); j < b.size(); ++j ){
                double val = stod(b[j].asString()); 
                bias_matrix(0,j) = val;    
        }

        cout << "Layer " << i << " " << " Weight \n" << endl;
        cout << weights_matrix << endl;
        
        cout << "Layer " << i << " " << " Bias \n" << endl;
        cout << bias_matrix << endl; 
    }  
 
}


Json::Value Network::ParseMatrix(MatrixXd w, MatrixXd b)
{
    Json::Value result;
    Json::Value weights(Json::arrayValue);
    Json::Value bias(Json::arrayValue);

    //Parse weights
    for (int i(0); i<w.rows();i++){
        Json::Value rows(Json::arrayValue);

        for (int j(0); j<w.cols();j++){
            rows.append(Json::Value(w(i,j)));
        }
        weights.append(rows);

    }

    //Parse bias
    for (int i(0); i<b.rows();i++){
        Json::Value rows(Json::arrayValue);

        for (int j(0); j<b.cols();j++){
                rows.append(Json::Value(b(i,j)));
        }
        bias.append(rows);
    }

    result["weights"] = weights ;
    result["bias"] = bias;

    return result;
};