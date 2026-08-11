#! /bin/bash

mkdir build
cd build
cmake ..
make

# install python package

cd ..
sudo python3 -m pip install .
