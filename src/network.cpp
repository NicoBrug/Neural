#include "../header/network.h"
#include "../header/fc_layer.h"


using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;

/** Constructor network with no argument
 * 
 */
Network::Network(){
    cout << "no specified network, if you wan't load network, please use constructor Network(string path)" << endl;

};

/** Constructor network with path of exist save network
 * 
 */
Network::Network(string s){
    Load(s);
};

/** Destructor -> liberate Layer memory
 * 
 */
Network::~Network(){
    for (int i(0); i< m_layer.size(); i++){
        delete(m_layer[i]);
    }
};

/** Adding a Layer to Network
 * 
 *  @param layer The pointer of the Layer Mother (class Layer)
 *  @return void
 * 
 */
void Network::Add(Layer *layer){
    m_layer.push_back(layer);
};

/** Predict data based on input data, forward propagation throughout the network 
 * 
 *  @param input_data Matrix Input data
 *  @return vector<MatrixXd> The array of Matrix output res
 * 
 */
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

/** Train the network on a set of data and a set of results, this is for set the good weights and bias
 * 
 *  @param x_train Matrix Input data
 *  @param y_train Matrix Result data
 *  @param epochs Number of iteration
 *  @param learning_rate The step size at each iteration
 *  @return void
 * 
 */
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


/** Mean Squared Error function, use to calculate the error between 2 data Matrix
 * 
 *  @param y_true Matrix Input data
 *  @param y_pred Matrix Result data
 *  @return double : error
 * 
 */
double Network::Mse(MatrixXd y_true, MatrixXd y_pred){
    MatrixXd diff = y_true-y_pred;
    return diff.array().pow(2).mean();
}

/** Derivative Mean Squared Error function, use to calculate the derivate of error between 2 data Matrix, use for retropropagation
 * 
 *  @param y_true Matrix Input data
 *  @param y_pred Matrix Pred data
 *  @return MatrixXd 
 * 
 */
MatrixXd Network::Mse_prime(MatrixXd y_true, MatrixXd y_pred){
    MatrixXd diff = y_pred-y_true;  
    return 2*diff/y_true.size();
}

/** Save network state in json file
 * 
 *  @param name String name of file we wan't to create
 *  @return true if Save 
 * 
 */
bool Network::Save(string name){
    file.open (name+".json");

    Json::Value event;   
    Json::Value vect(Json::arrayValue);

    for (int l(0); l< m_layer.size();l++){
        if (m_layer[l]->AsWeights()){
            MatrixXd weights = static_cast<Fc_Layer*>(m_layer[l])->GetWeights();
            MatrixXd bias = static_cast<Fc_Layer*>(m_layer[l])->GetBias();
            cout << "weights" << weights << endl;
            cout << "bias" << bias << endl;

            Json::Value json = Serialise(weights,bias);
            vect.append(json);

        }else{
            Json::Value result;
            result["type"] = "ActivationLayer" ;
            vect.append(result);

        }

    }
    event["Network"] = vect;

    file << event;

    file.close();

    return true;
}   

/** Load network state for json file, different type of Layer, Weight, Bias, and create Network from data loaded
 * 
 *  @param name String name of file we wan't to create
 *  @return void
 * 
 */
void Network::Load(string path){

    Json::Value root;
    std::ifstream file(path);
    file >> root;

    const Json::Value net = root["Network"];
     for ( int i(0); i < net.size(); ++i ){
        Json::Value Layer = net[i];

        if (Layer["type"] == "ActivationLayer"){
                Activation_layer* acl = new Activation_layer();
                this->Add(acl);
        }
        if (Layer["type"] == "FcLayer"){                
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

                Fc_Layer* fcl = new Fc_Layer(weights_matrix,bias_matrix);
                this->Add(fcl);
        }
    }  
}

/** Serialise bias and wheight to json
 * 
 *  @param w Matrix of weight we wan't to serialise
 *  @param b Matrix of bias we wan't to serialise
 *  @return Json::Value
 * 
 */
Json::Value Network::Serialise(MatrixXd w, MatrixXd b)
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
    result["type"] = "FcLayer" ;
    result["bias"] = bias;

    return result;
};