#!/bin/sh

echo "====================LOAD GCC MODULE========================"
module load compilers/gnu-5.4.0

echo "======================MAKE CLEAN==========================="
make clean

echo "=========================MAKE=============================="
make

echo "========================BENCH=============================="
python bench.py

echo "======================MAKE CLEAN==========================="
make clean