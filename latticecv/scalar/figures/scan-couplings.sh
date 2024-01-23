#!/bin/sh

L=20
scalar2d=../scalar2d
solved=../solved.py

for lambda in `seq 0.0 0.10 2.0`; do
	echo $lambda
	samples=data/scan-lambda-$lambda-samples.dat
	cor=data/scan-lambda-$lambda-cor.dat
	corlarge=data/scan-lambda-$lambda-cor-large.dat
	$scalar2d $L 0.1 $lambda 10000 > $samples
	$solved $L 0.1 $lambda < $samples > $cor
	$solved $L 0.1 $lambda --large-basis < $samples > $corlarge
done

