#!/bin/bash

set -e

for g in `seq 0.80 0.01 1.5`; do
		echo -n "$g "
		rm -rf data-$g
		./sample.jl -L 5 -g $g data-$g
		./observe.jl data-$g
done

