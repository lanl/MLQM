#!/usr/bin/env python

""" Sampling complex scalar field theory (in 0+1 dimensions). """

import sys

import numpy as np

import jax
import jax.numpy as jnp

from model import *

if True:
    def heatbath(key, phi, n):
        keyr, keyi, keyu = jax.random.split(key, 3)
        np = (n+1)%N
        nm = (n-1)%N
        phip = phi[np]
        phim = phi[nm]
        phi_ = phi[n] + (jax.random.normal(keyr) + 1j*jax.random.normal(keyi))
        u = jax.random.uniform(keyu)
        S = action_local(phim, phi[n], phip)
        Sp = action_local(phim, phi_, phip)
        acc = u < jnp.exp(S-Sp)
        return phi.at[n].set(acc*phi_ + (1-acc)*phi[n])

    @jax.jit
    def update(key, phi, K=1000):
        keyn, keyacc = jax.random.split(key, 2)
        keysacc = jax.random.split(keyacc, K)
        ns = jax.random.randint(keyn, (K,), 0, N)
        def hb(m, phi):
            return heatbath(keysacc[m], phi, ns[m])
        return jax.lax.fori_loop(0,K,hb,phi)

key = jax.random.PRNGKey(0)
phi = jnp.zeros(N) + 0j
while True:
    for _ in range(5):
        key, skey = jax.random.split(key)
        phi = update(skey,phi)
        phistr = ' '.join([str(x) for x in phi])
        print(phistr)

