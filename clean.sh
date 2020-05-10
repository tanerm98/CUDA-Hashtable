#!/bin/sh

make clean

rm out*
rm *.o*
rm *.e*
rm core*

git checkout -- .