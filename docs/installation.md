## Installation

Ok, first let's start by building the image (the development environment). 
```
xhost local:root
docker build -t neural-cuda .
```
Now we need to create the container. To do this we need to specify that we want to "share" the folder with our container. (path = The path to the folder where Neural is located )
```
sudo docker run \
    -it \
    --runtime=nvidia \
    --net=host \
    --privileged \
    -e DISPLAY=unix$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ~/Documents/neuralnetworks/Neural:/home/Projects/Neural \
    -t neural-cuda /bin/bash
```
Now we want to compile the project and check that the development environment is ready!
```
mkdir build
cd build
cmake ..
make
```