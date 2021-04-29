FROM nvidia/cuda:11.0-base

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -yq \
&& apt-get install cmake -y \
&& apt-get install build-essential -y \
&& apt-get install libjsoncpp-dev -y \
&& apt-get install pkg-config -y \
&& apt-get install libeigen3-dev -y \
&& apt-get install gnuplot -y \
&& apt-get install -qqy x11-apps -y\
&& apt-get install nvidia-cuda-dev -y \
&& apt-get install nvidia-cuda-toolkit -y\
&& apt-get install doxygen -y \
&& apt-get update -y \
&& apt-get clean




WORKDIR /home/Projects/Neural 