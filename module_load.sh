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
./gpu_hashtable 4000000 2

echo "========================BENCH=============================="
python bench.py

echo "======================MAKE CLEAN==========================="
make clean