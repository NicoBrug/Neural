#include "../header/network.h"
#include "../header/fc_layer.h"

using namespace std;
using namespace std::chrono;
using Eigen::MatrixXd;
using namespace Eigen;
using namespace sciplot;

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

/** Adding a loss function to network
 * 
 *  @param loss The pointer of the loss function
 *  @return void
 * 
 */
void Network::Use(Loss *l){
    m_loss = l;
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

    auto start = high_resolution_clock::now();
    
    for (int i(0);i<epochs; i++){
        double err(0);

        for (int j(0); j<samples; j++){
            
            MatrixXd output(1,cols);
            output = x_train.row(j);
            

            for (int l(0);l<m_layer.size();l++){
                output = m_layer[l]->Forward_propagation(output);
            } 

            err += this->m_loss->Compute(y_train.row(j), output);
            MatrixXd error = this->m_loss->Compute_prime(y_train.row(j),output);

            for (int k(m_layer.size()-1); k>=0; k--){
                error = m_layer[k]->Backward_propagation(error,learning_rate);
            }   
        }

        //cout << "epoch " << i+1 << " | " << "error " << err/samples << endl;
        m_error.push_back(err/samples); // for plotting
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    PlotData(epochs,m_error);

    cout << "\nTime taken by fitting: "
         << duration.count() << " microseconds" << endl;
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
        Json::Value json  = static_cast<Fc_Layer*>(m_layer[l])->toJSON();
        vect.append(json);
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
                cout << "layer" << i << endl;

                for ( int j(0); j < weights.size(); ++j ){
                    Json::Value w = weights[j];
                    for ( int k(0); k < w.size(); ++k ){
                        double val = stod(w[k].asString()); 
                        weights_matrix(j,k) = val;   
                    }
                }
                Json::Value b = bias[0];
                //cout << "weights" << weights_matrix << endl;

                for ( int j(0); j < b.size(); ++j ){
                        double val = stod(b[j].asString()); 
                        bias_matrix(0,j) = val;    
                }
                //cout <<  "bias" << bias_matrix << endl;

                Fc_Layer* fcl = new Fc_Layer(weights_matrix,bias_matrix);
                this->Add(fcl);
        }
    }  
}


void Network::PlotData(int epochs, vector<double> error){
    Plot plot;

    Vec time = linspace(0.0, epochs, epochs);
    
    plot.xlabel("iteration");
    plot.ylabel("y");

    plot.xrange(0.0, epochs);
    plot.yrange(0.0, 1);

    plot.drawCurve(time, error).label("error");
    plot.show();
    plot.save("error.pdf"); 
}