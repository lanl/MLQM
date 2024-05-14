#!/usr/bin/env python

import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jr

def K_component(m, A, t,x, tp,xp):
    L = A.shape[0]
    def diag(t,x):
        return m + 0j

    def t_p(t,x):
        return -1/2 * jnp.exp(- 1j*A[t,x,0])

    def t_m(t,x):
        return 1/2 * jnp.exp(+ 1j*A[t,x,0])

    def x_p(t,x):
        return -1/2 * jnp.exp(-1j*A[t,x,1])

    def x_m(t,x):
        return 1/2 * jnp.exp(1j*A[t,x,1])

    def nada(t,x):
        return 0.j

    dt = (tp-t)%L
    dx = (xp-x)%L

    ret = 0.j
    ret += jax.lax.cond(jnp.logical_and(dt==0, dx==0),diag,nada,t,x)
    ret += jax.lax.cond(jnp.logical_and(dt==1, dx==0),t_p,nada,t,x)
    ret += jax.lax.cond(jnp.logical_and(dt==-1%L, dx==0),t_m,nada,tp,x)
    ret += jax.lax.cond(jnp.logical_and(dt==0, dx==1),x_p,nada,t,x)
    ret += jax.lax.cond(jnp.logical_and(dt==0, dx==-1%L),x_m,nada,t,xp)

    return ret

def K(m, A):
    L = A.shape[0]
    t, x = jnp.indices((L, L))
    t, x = t.ravel(), x.ravel()
    return jax.vmap(lambda tp,xp: jax.vmap(lambda t,x: K_component(m, A,tp,xp,t,x))(t,x))(t,x)

def logdetK(m, A):
    return jnp.linalg.slogdet(K(m, A))[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Generate training data",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            fromfile_prefix_chars='@'
            )
    parser.add_argument('--debug-nan', action='store_true',
            help='nan debugging mode')
    parser.add_argument('--debug-inf', action='store_true',
            help='inf debugging mode')
    parser.add_argument('--no-jit', action='store_true',
            help='disable JIT compilation')
    parser.add_argument('--x64', action='store_true',
            help='64-bit mode')
    parser.add_argument('--seed', type=int, 
            help="random seed")
    parser.add_argument('N', type=int, help="number of samples to generate")
    parser.add_argument('L', type=int, help="lattice size")
    parser.add_argument('T', type=float, help="temperature")
    args = parser.parse_args()

    if args.debug_nan:
        jax.config.update("jax_debug_nans", True)
    if args.debug_inf:
        jax.config.update("jax_debug_infs", True)
    if args.no_jit:
        jax.config.update("jax_disable_jit", True)
    if args.x64:
        jax.config.update("jax_enable_x64", True)

    # PRNG
    if args.seed is None:
        seed = time.time_ns()
    else:
        seed = args.seed
    key = jr.PRNGKey(seed)

    @jax.jit
    def generate(k, T):
        A = T*jr.normal(k, (args.L,args.L,2))
        ld = logdetK(0., A)
        return ld, A

    for n in range(args.N):
        gk, key = jr.split(key)
        d, A = generate(key, args.T)
        Astr = ' '.join(map(str, A.ravel()))
        print(d, Astr)

