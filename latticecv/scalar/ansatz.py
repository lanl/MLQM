#!/usr/bin/env python

# Optimize a subtraction in a small family


import argparse
import sys
import time
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg,gmres
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from model import *
from util import *

class NoCV(eqx.Module):
    def __init__(self, action):
        pass

    def __call__(self, phi):
        return 0.

class FirstOrderCV(eqx.Module):
    action: Callable
    m: jnp.array
    b: jnp.array

    def __init__(self, action):
        self.action = action
        self.m = jnp.array(1.)
        self.b = jnp.array([1.])

    def __call__(self, phi):
        def coef(x,y):
            x2 = 1 - jnp.cos(2*np.pi*x/L)
            y2 = 1 - jnp.cos(2*np.pi*y/L)
            r = jnp.sqrt(x2 + y2)
            return self.b[0] * jnp.exp(-self.m * r)

        if False:
            def term(x, y, xp, yp):
                dx = xp-x
                dy = yp-y
                c = coef(x,y)
                return 0.

            def term_(idx):
                x = idx%L
                idx = idx//L
                y = idx%L
                idx = idx//L
                xp = idx%L
                idx = idx//L
                yp = idx%L
                return term(x, y, xp, yp)

            return jnp.sum(jax.lax.map(term_, jnp.arange(L**4)))

        if True:
            c = jnp.vectorize(coef)(*jnp.indices((L,L)))
            def at(x,y):
                # Compute subtraction using (x,y) as a base point.
                phi_ = jnp.roll(phi, (x,y))
                def f(pi):
                    phip = phi_.at[0,0].add(pi)
                    return jnp.mean(c*phip)
                def S(pi):
                    phip = phi_.at[0,0].add(pi)
                    return self.action(phip)
                df = jax.grad(f)(0.)
                dS = jax.grad(S)(0.)
                return df - f(0.)*dS
            def at_(xy):
                x = xy%L
                y = xy//L
                return at(x,y)
            return jnp.mean(jax.lax.map(at_, jnp.arange(L*L)))

class ThirdOrderCV(eqx.Module):
    def __init__(self, action):
        pass

    def __call__(self, phi):
        def coef1(x,y):
            return 0.
        c1 = jnp.vectorize(coef1)(*jnp.indices((L,L)))
        def coef3(x,y):
            return 0.
        c3 = jnp.vectorize(coef3)(*jnp.indices((L,L)))
        # TODO
        return 0.

class ThreePointCV(eqx.Module):
    def __init__(self, action):
        pass

    def __call__(self, phi):
        def coef1(x,y):
            return 0.
        c1 = jnp.vectorize(coef1)(*jnp.indices((L,L)))
        def coef3(x,y):
            return 0.
        c3 = jnp.vectorize(coef3)(*jnp.indices((L,L)))
        # TODO
        return 0.

CVs = [NoCV, FirstOrderCV, ThirdOrderCV, ThreePointCV]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Obtain contour",
            )
    parser.add_argument('--verbose', action='store_true',
            help="Verbose output")
    parser.add_argument('--plot', action='store_true',
            help='Produce plot')
    parser.add_argument('--level', type=int, required=True,
            help='Size of CV to use')
    parser.add_argument('--steps', type=int, default=500,
            help='Number of training steps to take')
    parser.add_argument('L', type=int)
    parser.add_argument('m2', type=float)
    parser.add_argument('lamda', type=float)
    args = parser.parse_args()

    model = Model(args.L, args.m2, args.lamda)
    L = model.L
    N = L**2

    CV = CVs[args.level]

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

    cv = CV(model.action)

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

