#!/bin/sh

echo "====================LOAD GCC MODULE========================"
module load compilers/gnu-5.4.0

echo "====================LOAD CUDA MODULE======================="
module load libraries/cuda

echo "======================MAKE CLEAN==========================="
make clean

echo "=========================MAKE=============================="
make

echo "========================DEBUG=============================="
./gpu_hashtable 1000000 1

echo "========================BENCH=============================="
python3.5 bench.py

echo "======================MAKE CLEAN==========================="
make clean