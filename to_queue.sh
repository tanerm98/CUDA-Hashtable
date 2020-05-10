#!/bin/sh

qsub -q hp-sl.q -b y -wd ~/CUDA-Hashtable ./module_load.sh
#qsub -q ibm-dp.q -b y -wd ~/CUDA-Hashtable ./module_load.sh
qstat