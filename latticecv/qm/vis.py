#!/usr/bin/env python

import argparse
from functools import partial
import sys

import progressbar
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from model import *

from linear import *
from ansatz import *
from gd import *

from util import *

matplotlib.style.use('classic')
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.prop_cycle'] = plt.cycler(color='krbg')
matplotlib.rcParams['legend.numpoints'] = 1

phi = np.array([[complex(x) for x in l.split()] for l in sys.stdin.readlines()])
K = phi.shape[0]

# Compute actions.
S = np.vectorize(action, signature='(i)->()')(phi)
dS = np.vectorize(grad_action, signature='(i)->(j)')(phi)

ts = np.arange(N)
cor_raw = np.zeros(N, dtype=np.complex128)
sig_raw = np.zeros(N, dtype=np.complex128)
cor_vr = np.zeros(N, dtype=np.complex128)
sig_vr = np.zeros(N, dtype=np.complex128)

def correlator(phi, n):
    return phi[0].conj() * phi[n]
correlator_v = jax.jit(jax.vmap(correlator, in_axes=(0,None)))

phi = jnp.array(phi)
for n in progressbar.progressbar(ts):
    print(n)
    cor = correlator_v(phi, n)
    cor_raw[n], sig_raw[n] = bootstrap(cor)
    vr = jax.vmap(linear_vr(lambda x: correlator(x, n), phi))(phi)
    cor_vr[n], sig_vr[n] = bootstrap(vr)

plt.figure(figsize=(5,4), dpi=600)
plt.errorbar(ts-0.1, cor_raw.real, yerr=sig_raw.real, fmt='.')
plt.errorbar(ts+0.1, cor_vr.real, yerr=sig_vr.real, fmt='.')
plt.xlabel('$\\tau$')
plt.ylabel('$\\langle \\phi(\\tau) \\phi(0) \\rangle$')
plt.yscale('log')
plt.xlim([0,N])
plt.tight_layout()
plt.savefig('fig.png')
#plt.show()

