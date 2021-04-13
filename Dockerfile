FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -yq \
&& apt-get install cmake -y \
&& apt-get install build-essential -y \
&& apt-get install libjsoncpp-dev -y \
&& apt-get install pkg-config -y \
&& apt-get install libeigen3-dev -y \
&& apt-get install gnuplot -y \
&& apt-get install -qqy x11-apps -y\
&& apt install libnvidia-gl-440 \
&& apt-get update -y \
&& apt-get clean

WORKDIR /home/Projects/Neural 