#!/bin/bash

# Store initial number of points (i.e. size of LHS samp,le)
num_points=$(python -u main.py start_samp)
echo 'Size of initial training set =' $num_points | tee output

# Modify 'create_lhs.R' to create 'lhs_samp.dat' with correct number of points
sed -i "s/n_points = [0-9]*/n_points = $num_points/" create_lhs.R

# Generate 'lhs_samp.dat'
Rscript create_lhs.R

# Run program in normal mode, store all printed output in 'output'
python -u main.py iterative |& tee -a output

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
