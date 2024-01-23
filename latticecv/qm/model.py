import sys

import jax
import jax.numpy as jnp

def load_model(filename):
    global N,mass,lamda
    v = {}
    with open(filename) as f:
        exec(f.read(), None, v)
    N = v['N']
    mass = v['mass']
    lamda = v['lamda']

load_model(sys.argv[1])

def action_local(phim, phi, phip):
    return (jnp.abs(phip-phi)**2 + jnp.abs(phim-phi)**2) / 2. + mass**2/2. * jnp.abs(phi)**2 + lamda/24. * jnp.abs(phi)**4

@jax.jit
def action(phi):
    phip = jnp.roll(phi, 1)
    kin = jnp.sum(jnp.abs(phi-phip)**2)/2.
    pot = jnp.sum(mass**2/2. * jnp.abs(phi)**2 + lamda/24. * jnp.abs(phi)**4)
    return kin + pot

def action_ri(phir, phii):
    return action(phir+1j*phii)

@jax.jit
def grad_action(phi):
    phir, phii = phi.real, phi.imag
    gr, gi = jax.grad(action_ri, argnums=(0,1))(phir, phii)
    return jnp.concatenate([gr,gi])

