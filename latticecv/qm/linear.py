# Control variate defined by linear regression.

import jax
import jax.numpy as jnp

import numpy as np

from model import *

@jax.jit
def _basis(phi):
    """
    Compute a bunch of observables, each with expectation value guaranteed to
    vanish.
    """
    def f_(k, imag, g, phi):
        # The derivative is with respect to the kth component; the real part if
        # imag is False, and the imaginary part if True.
        I = 1j if imag else 1
        def g_(x):
            phi_ = phi.at[k].add(x*I)
            return g(phi_)
        def S_(x):
            phi_ = phi.at[k].add(x*I)
            return action(phi_)
        g_r = lambda x: g_(x).real
        g_i = lambda x: g_(x).imag
        S_r = lambda x: S_(x).real
        S_i = lambda x: S_(x).imag

        dg = jax.grad(g_r)(0.) + 1j*jax.grad(g_i)(0.)
        dS = jax.grad(S_r)(0.) + 1j*jax.grad(S_i)(0.)

        return dg - g(phi)*dS

    def g(phi):
        return phi[1]

    def f1(i, j, imag, phi):
        def g(phi):
            return phi[i]
        return f_(j, imag, g, phi)

    #fs_r = jnp.ravel(jnp.vectorize(lambda i,j: f1(i,j,False,phi))(*jnp.indices((N,N))))
    #fs_i = jnp.ravel(jnp.vectorize(lambda i,j: f1(i,j,True,phi))(*jnp.indices((N,N))))
    fs_r = jnp.ravel(jnp.vectorize(lambda i,j: f1(i,j,False,phi))(*jnp.indices((N,N))))
    fs_i = jnp.ravel(jnp.vectorize(lambda i,j: f1(i,j,True,phi))(*jnp.indices((N,N))))
    return jnp.concatenate([fs_r, fs_i])

@jax.jit
def _solve(obs, phi):
    """

    Returns a list of coefficients.
    """
    # Potential control variates.
    f = jax.vmap(_basis)(phi)

    N = f.shape[1]

    def M_(ij):
        i = ij // N
        j = ij % N
        return jnp.mean(f[:,i].conj() * f[:,j])
    M = jax.lax.map(M_, jnp.arange(N*N)).reshape((N,N))

    def v_(i):
        return jnp.mean(f[:,i].conj() * obs)
    v = jax.lax.map(v_, jnp.arange(N))

    if False:
        # Compute the covariance matrix.
        def M_(i,j):
            return jnp.mean(f[:,i].conj() * f[:,j])
        M = jnp.vectorize(M_)(*jnp.indices((N,N)))

        # Compute correlations with target observable.
        def v_(i):
            return jnp.mean(f[:,i].conj() * obs)
        v = jax.vmap(v_)(*jnp.indices((N,)))

    return jnp.linalg.solve(M, v)

def linear_cv(obs, phi):
    """
    Takes in a list of samples, and a target observable, and constructs an
    optimal control variate.

    Returns a function for computing the control variate.
    """
    c = _solve(obs, phi)
    return lambda x: jnp.einsum('i,i->', c, _basis(x))

def linear_vr(obs, phi):
    """
    Takes in a list of samples, and a target observable, and constructs a
    variance-reduced observable.

    Returns a function for computing the variance-reduced observable.
    """
    cv = linear_cv(jax.vmap(obs)(phi), phi)
    return lambda x: obs(x) - cv(x)

