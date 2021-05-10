FROM nvidia/cudagl:11.2.1-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -yq \
&& apt-get install build-essential -y \
&& apt-get install libjsoncpp-dev -y \
&& apt-get install pkg-config -y \
&& apt-get install libeigen3-dev -y \
&& apt-get install gnuplot -y \
&& apt-get install libssl-dev \
&& apt-get install doxygen -y \
&& apt install texlive-latex-base -y \
&& apt-get update -y \
&& apt-get clean


WORKDIR /home/Projects/Neural 