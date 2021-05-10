\page InstallationPage Installation
This section is dedicated to developers who want to build the application from source.

\section InstallationPageSec1  Prerequisites

* Docker

The only prerequisite is to have docker, for the rest, we will install all the necessary software. Please follow the instructions precisely, and don't hesitate to report any bug. 
First of all, make a clone of Neural. 
```
git clone https://github.com/NicoBrug/Neural.git
```
\section InstallationPageSec2  Docker

Ok, first let's start by get the image (the development environment). 
```
docker push nickoslab/neural:alpha
```
Now we need to create the container. To do this we need to specify that we want to "share" the folder with our container. (path = The path to the folder where Neural is located )
```
xhost local:root
sudo docker run \
    -it \
    --runtime=nvidia \
    --net=host \
    --privileged \
    -e DISPLAY=unix$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/Documents/neuralnetworks/Neural:/home/Projects/Neural \
    --name NeuralGL \
    -t neural:alpha /bin/bash 
```
\section InstallationPageSec3  Nvidia & Cuda
Once the container is launched, you can check the installation of the drivers with : 
```
xhost local:root
nvidia-smi
```
Now you have to install the nvidia toolkit for development with cuda. Since the drivers are already configured, you don't need to reinstall them. So you just have to get the nvidia toolkit installer and disable the drivers installation. /!\ If you don't disable the drivers installation, it could create a conflict with the existing drivers. 
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sh cuda_11.3.0_465.19.01_linux.run
```
\section InstallationPageSec4  Cmake
To install the right version of cmake, please go to the official website and download the corresponding version (>18.6). 
https://cmake.org/download/
First, remove the bad version of cmake 
```
apt remove cmake
hash -r
```
Go to the cmake folder and type :
```
./configure
make
make install
```
\section InstallationPageSec5  Qt
For the installation of Qt, you can use the online installer. If you use another version of Qt (not 5.15.2) you have to specify it in the CMakeLists.txt. (don't forget to check Qt3D)
```
export LD_LIBRARY_PATH=/opt/Qt/5.15.2/gcc_64/lib
```
\section InstallationPageSec6  Build
Now we want to compile the project and check that the development environment is ready!
```
mkdir build
cd build
cmake ..
make
```
Everything to compile without problems? Then you can go to the next step, run the XOR example.
```
./test-neural
```