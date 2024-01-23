#!/usr/bin/env python

# Visualize optimal subtractions

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import cg,gmres
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import *
from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Obtain contour",
            )
    parser.add_argument('--large-basis', action='store_true',
            help='Use large basis')
    parser.add_argument('L', type=int)
    parser.add_argument('m2', type=float)
    parser.add_argument('lamda', type=float)
    args = parser.parse_args()

    model = Model(args.L, args.m2, args.lamda)
    L = model.L
    N = L**2

    phi = jnp.array([[float(x) for x in l.split()] for l in sys.stdin.readlines()])
    K = phi.shape[0]

    # On each sample, we need to know the gradient of the action.
    dS = jax.vmap(jax.grad(model.action))(phi)

    phimean = jnp.mean(phi.reshape((K,L,L)),axis=2)

    dts = np.arange(L)
    cor_raw = np.zeros(L)
    sig_raw = np.zeros(L)
    cor_vr = np.zeros(L)
    sig_vr = np.zeros(L)

    # Block-averaging.
    def block(dat):
        # We will convert dat into Nblocks blocks.
        S = dat.shape[0]//Nblocks
        # Each block will be of size S. Truncate:
        dat = dat[:S*Nblocks]
        # Reshape so that we can average over an axis.
        dat = dat.reshape((Nblocks,S)+dat.shape[1:])
        return jnp.mean(dat, axis=1)

    # Compute possible control variates.
    def basis_(phi, dS):
        phi = phi.reshape((K,model.L*model.L))
        dS = dS.reshape((K,model.L*model.L))
        def f_(phi):
            # Functions used to generate S-D relations.
            #return jnp.array([1.,phi[0]])
            if args.large_basis:
                return jnp.array([phi[0], phi[0]**3])
            else:
                return jnp.array([phi[0]])
        f = jax.vmap(f_)(phi)
        Nfs = f.shape[1]
        df = jax.vmap(jax.jacfwd(f_))(phi)
        # f.shape is (K,Nfs); df.shape is (K,Nfs,N); dS.shape is (K,N)
        sub = df - jnp.einsum('ki,kj->kij', f, dS)
        # sub.shape is (K,Nfs,N)
        return sub.reshape((K,N*Nfs))
    basis = 0*basis_(phi, dS)
    phi_ = phi.reshape((K,model.L,model.L))
    dS_ = dS.reshape((K,model.L,model.L))
    for x in range(model.L):
        for y in range(model.L):
            phi_t = jnp.roll(phi_, (x,y), axis=(1,2))
            dS_t = jnp.roll(dS_, (x,y), axis=(1,2))
            basis = basis + basis_(phi_t, dS_t)

    B = basis.shape[1]

    # Number of samples reserved for use in fitting.
    Nblocks = 8*B
    BT = 8*B


    # Block
    bbasis = block(basis)

    if True:
        for i in range(B):
            m, e = bootstrap(bbasis[:,i])
            if np.abs(m) > 3*np.abs(e):
                print(i,m,e,file=sys.stderr)

    # Check that we're not overfitting
    print(f'{B} basis functions, training on {BT} blocks', file=sys.stderr)

    for dt in tqdm(dts):
        # Get raw correlator.
        phiroll = jnp.roll(phimean, dt, axis=1)
        phicorr = jnp.mean(phimean.conj()*phiroll, axis=1).real
        bphicorr = block(phicorr)
        
        # Get variance-reduced correlator.
        def M_(ij):
            i = ij // B
            j = ij % B
            bi = bbasis[:BT,i]
            bj = bbasis[:BT,j]
            return jnp.mean(bi * bj) - jnp.mean(bi)*jnp.mean(bj)
        M = jax.lax.map(M_, jnp.arange(B*B)).reshape((B,B))

        def v_(i):
            return jnp.mean((bbasis[:BT,i]-jnp.mean(bbasis[:BT,i])) * bphicorr[:BT])
        v = jax.lax.map(v_, jnp.arange(B))

        c = jnp.linalg.solve(M, v)
        c = c.reshape((L,L))
        d = jnp.fft.rfft2(c)
        print(d)
        #c = np.roll(c, (L//2,L//2))
        if True:
            extreme = jnp.max(jnp.abs(c))
            plt.imshow(c, cmap='seismic', vmin=-extreme, vmax=extreme)
        else:
            extreme = jnp.max(jnp.abs(d))
            plt.imshow(d.imag, cmap='seismic', vmin=-extreme, vmax=extreme)
        plt.show()

