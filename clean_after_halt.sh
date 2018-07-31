#!/bin/bash

# Organise plots and data
mkdir run$1
mv output run$1
mv *.pdf run$1
mv *.dat run$1

cd run$1

mkdir min_data
mv Z_*.dat min_data

mkdir plots
mv *.pdf plots

mkdir emul_data
mv emul*.dat LML*.dat emul_data
mv diff*.dat conf*.dat training_set*.dat y_pred*.dat x_pred.dat emul_data

cd ..
