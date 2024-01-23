#!/bin/sh

# Scanning on a small system to get precise results, as a test of correctness.

scalar2d=../scalar2d
solved=../solved.py
raw=../raw.py

L=24
m2=0.
lambda=2.0

# Get samples
$scalar2d $L $m2 $lambda 10000 > data/large.dat
$solved $L $m2 $lambda < data/large.dat > data/large-solved.dat
$solved $L $m2 $lambda --large-basis < data/large.dat > data/large-solved-large.dat
$raw $L $m2 $lambda < data/large.dat > data/large-raw.dat

