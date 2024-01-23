#!/usr/bin/env python

# Plot raw correlator---no control variate applied.

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from model import *
from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Obtain contour",
            )
    parser.add_argument('--plot', action='store_true',
            help='Produce plot')
    parser.add_argument('L', type=int)
    parser.add_argument('m2', type=float)
    parser.add_argument('lamda', type=float)
    args = parser.parse_args()

    model = Model(args.L, args.m2, args.lamda)
    N = model.L * model.L

    phi = jnp.array([[complex(x) for x in l.split()] for l in sys.stdin.readlines()])
    K = phi.shape[0]
    phi = phi.reshape((K, model.L, model.L))

    if args.plot:
        plt.plot(phi[:,0,0].real)
        plt.show()

    # Get means
    phimean = jnp.mean(phi,axis=2)

    cor, err = np.zeros(model.L), np.zeros(model.L)

    dts = np.array(range(model.L))
    for dt in dts:
        phiroll = jnp.roll(phimean, dt, axis=1)
        phicorr = jnp.mean(phimean.conj()*phiroll, axis=1)
        c, e = bootstrap(phicorr)
        cor[dt] = complex(c).real
        err[dt] = complex(e).real

    if args.plot:
        plt.errorbar(dts, cor, yerr=err)
        plt.show()

    else:
        for dt in dts:
            print(f'{dt} {cor[dt]} {err[dt]}')

