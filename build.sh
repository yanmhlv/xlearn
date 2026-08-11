#!/bin/bash

mkdir build
cd build
cmake ..
make

# install python package
# The wheel builds the library it needs itself, so it is installed from the
# project root rather than from the tree the command line tools were built in.

cd ..
python3 -m pip install .
