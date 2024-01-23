#!/usr/bin/env python

# Learn a neural network


import argparse
import sys
import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from model import *
from util import *

class CV(eqx.Module):
    mlp: eqx.nn.MLP
    action: Callable
    V: int

    def __init__(self, V, action, *, key, width_scale, depth):
        self.mlp = eqx.nn.MLP(
                in_size = V,
                out_size = V,
                width_size = V*width_scale,
                depth = depth,
                activation = jax.nn.relu,
                key = key
        )
        self.action = action
        self.V = int(V)

    def __call__(self, phi):
        phi = phi.reshape(self.V)
        f = self.mlp(phi)
        df = jax.jacfwd(self.mlp)(phi)
        dS = jax.jacfwd(self.action)(phi)
        return jnp.sum(df - f*dS)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Obtain contour",
            )
    parser.add_argument('--verbose', action='store_true',
            help="Verbose output")
    parser.add_argument('--plot', action='store_true',
            help='Produce plot')
    parser.add_argument('--steps', type=int, default=500,
            help='Number of training steps to take')
    parser.add_argument('L', type=int)
    parser.add_argument('m2', type=float)
    parser.add_argument('lamda', type=float)
    args = parser.parse_args()

    model = Model(args.L, args.m2, args.lamda)
    L = model.L
    N = L**2

    phi = jnp.array([[float(x) for x in l.split()] for l in sys.stdin.readlines()[::10]])
    K = phi.shape[0]
    phi = phi.reshape((K,L,L))

    phi_t = phi[:K//2]
    phi_d = phi[K//2:]

    # Destination for results
    dts = np.arange(L)
    cors = np.zeros(L)
    errs = np.zeros(L)
    cors_raw = np.zeros(L)
    errs_raw = np.zeros(L)

    cv = CV(L*L, model.action, key=jr.PRNGKey(0), width_scale=2, depth=2)

    def correlator(phi, dt):
        phimean = jnp.mean(phi.reshape((L,L)), axis=1)
        phiroll = jnp.roll(phimean, dt)
        phicorr = jnp.mean(phimean*phiroll)
        return phicorr

    def loss(cv, obs):
        return jnp.std(obs - jax.lax.map(cv,phi_t))

    @eqx.filter_jit
    def step(cv, opt_state, obs):
        lval, grad = eqx.filter_value_and_grad(loss)(cv, obs)
        updates, opt_state = opt.update(grad, opt_state)
        cv = eqx.apply_updates(cv, updates)
        return cv, opt_state, lval

    for dt in tqdm(dts):
        opt = optax.adam(3e-2)
        opt_state = opt.init(eqx.filter(cv, eqx.is_array))
        obs = jax.lax.map(lambda p: correlator(p, dt), phi_t)
        for k in range(args.steps):
            cv, opt_state, lval = step(cv, opt_state, obs)
            if args.verbose:
                print(f'{k} {lval}')

        obs = jax.lax.map(lambda p: correlator(p, dt), phi_d)
        cor_raw, err_raw = bootstrap(obs)
        cors_raw[dt], errs_raw[dt] = cor_raw, err_raw
        obs = obs - jax.lax.map(cv, phi_d)
        cor, err = bootstrap(obs)
        cors[dt], errs[dt] = cor, err
        if args.verbose:
            print(cor, err, cor_raw, err_raw)

    if args.plot:
        plt.errorbar(dts, cors, yerr=errs, fmt='.')
        plt.xlabel('$\\tau$')
        plt.ylabel('$\\langle \\phi(\\tau) \\phi(0) \\rangle$')
        plt.yscale('log')
        plt.xlim([-0.5,L+.5])
        plt.show()
    else:
        for dt in dts:
            print(f'{dt} {cors[dt]} {errs[dt]}')

