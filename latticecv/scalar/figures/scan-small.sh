#!/bin/sh

# Scanning on a small system to get precise results, as a test of correctness.

scalar2d=../scalar2d
solved=../solved.py
raw=../raw.py

# Get samples
$scalar2d 8 0.1 0.5 1000 > data/small-short.dat
$scalar2d 8 0.1 0.5 100000 > data/small-large.dat
$solved 8 0.1 0.5 < data/small-short.dat > data/small-solved.dat
$raw 8 0.1 0.5 < data/small-large.dat > data/small-raw.dat

