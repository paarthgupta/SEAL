#!/bin/bash

cd ../../
git clone https://github.com/muhanzhang/pytorch_DGCNN
cd pytorch_DGCNN
cd lib
make -j4
cd "$(dirname "$0")"
pip install numpy
pip install scipy
pip install networkx
pip install tqdm
pip install sklearn
pip install gensim
