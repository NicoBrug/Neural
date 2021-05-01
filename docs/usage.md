## Usage

Create the network
```
Network net; 
```
or if we already save a Neural network :
```
Network net("../net.json"); 
```
Do you want to assign a number of threads to optimize the operation? 
```
net.SetThread(5); 
```
If you don't know how much core your pc has -> 
```
cout << "my cores: " << net.GetThreads << endl;
```
Instantiate the different activation function :
```
Activation* than = new Than();
Activation* sigmoid = new Sigmoid();
Activation* relu = new Relu();
Activation* softplus = new SoftPlus();
```
Instantiate the loss function & use it in Network
```
Loss* mse = new Mse();
Loss* cre = new Cross_entropy();
net.Use(mse);
```
Instantiate the different Layer : Fc_Layer = full connected neuron
```
Fc_Layer* fcl1 = new Fc_Layer(2,5);
Activation_layer* acl1 = new Activation_layer(than);
```
Add the different layer to Network :
```
net.Add(fcl1);
net.Add(acl1);
```
Train the network : 
```
net.Fit(x_data,x_train,1000,0.1);
```
Use the network :
```
net.Predict(x_test);
```
Save :
```
net.Save("My_Amazing_Weights");
```
Plot:
```
net.Plot("loss");
```
