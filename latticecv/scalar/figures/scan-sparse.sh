#!/bin/sh

# A huge lattice and a sparse CV

scalar2d=../scalar2d
sparse=../sparse.py

L=50
m2=0.1
lambda=0.1
mu=1e-4

# Get samples
$scalar2d $L $m2 $lambda 1000 > data/huge.dat
$sparse $L $m2 $lambda $mu < data/huge.dat > data/huge-solved.dat

