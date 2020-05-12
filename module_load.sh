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
time ./gpu_hashtable 1000 2

echo "========================NVPROF============================="
nvprof ./gpu_hashtable 2000000 2

#echo "========================BENCH=============================="
#python bench.py

echo "======================MAKE CLEAN==========================="
make clean