#!/usr/bin/env python

import sys
import numpy as np
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
matplotlib.style.use('classic')
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['axes.prop_cycle'] = plt.cycler(color='krbg')
matplotlib.rcParams['legend.numpoints'] = 1

def read_data(fn, dtype=np.float64):
    with open(fn) as f:
        return np.array([[dtype(x) for x in l.split()] for l in f.readlines()])


# SMALL

dat_cv = read_data('data/small-solved.dat')
dat_raw = read_data('data/small-raw.dat')

plt.figure(figsize=(6,4), dpi=500)
plt.errorbar(dat_cv[:,0]-1e-1, dat_cv[:,1], yerr=dat_cv[:,2], fmt='.', label='Raw, $10^3$ samples')
plt.errorbar(dat_cv[:,0], dat_cv[:,3], yerr=dat_cv[:,4], fmt='.', label='Variance reduced, $10^3$ samples')
plt.errorbar(dat_raw[:,0]+1e-1, dat_raw[:,1], yerr=dat_raw[:,2], fmt='.', label='Raw, $10^5$ samples')
plt.xlabel('$\\tau$')
plt.ylabel('$\\langle\\phi(\\tau)\\phi(0)\\rangle$')
plt.yscale('log')
plt.xlim([-0.5,7.5])
plt.ylim([3e-2,3e-1])
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('small.png')

# BIG LATTICE

dat_cv_small = read_data('data/large-solved.dat')
dat_cv_large = read_data('data/large-solved-large.dat')
plt.figure(figsize=(6,4), dpi=500)
plt.errorbar(dat_cv_small[:,0]-2e-1, dat_cv_small[:,3], yerr=dat_cv_small[:,4], fmt='.', label='Variance reduced')
plt.errorbar(dat_cv_small[:,0], dat_cv_small[:,1], yerr=dat_cv_small[:,2], fmt='.', label='Raw')
plt.errorbar(dat_cv_large[:,0]+2e-1, dat_cv_large[:,3], yerr=dat_cv_large[:,4], fmt='.', label='Variance reduced, large basis')
plt.xlabel('$\\tau$')
plt.ylabel('$\\langle\\phi(\\tau)\\phi(0)\\rangle$')
plt.yscale('log')
plt.xlim([-0.5,24-.5])
#plt.ylim([3e-2,3e-1])
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('large.png')

# COUPLING SCAN

lamdas = np.arange(0,2.01,.10)
#lamdas = np.arange(0,1.01,.05)
cor0 = np.zeros_like(lamdas)
err0 = np.zeros_like(lamdas)
cor1 = np.zeros_like(lamdas)
err1 = np.zeros_like(lamdas)
cor2 = np.zeros_like(lamdas)
err2 = np.zeros_like(lamdas)
tau = 10
for i, lamda in enumerate(lamdas):
    lstr = f'{lamda:.2f}'
    dat = read_data(f'data/scan-lambda-{lstr}-cor.dat')
    cor0[i] = dat[tau,1]
    err0[i] = dat[tau,2]
    cor1[i] = dat[tau,3]
    err1[i] = dat[tau,4]
    dat = read_data(f'data/scan-lambda-{lstr}-cor-large.dat')
    cor2[i] = dat[tau,3]
    err2[i] = dat[tau,4]
ok0 = cor0/err0 > 2
fig = plt.figure(figsize=(6,4), dpi=500)
ax = fig.subplots()
ax.errorbar(lamdas[ok0]-.02, cor0[ok0], yerr=err0[ok0], fmt='.', label='No VR', color='black')
ax.errorbar(lamdas, cor1, yerr=err1, fmt='.', label='First-order', color='red')
ax.errorbar(lamdas+.02, cor2, yerr=err2, fmt='.', label='Third-order', color='blue')
ax.set_yscale('log')
ax.set_ylabel('$\\langle\\phi(\\tau)\\phi(0)\\rangle|_{\\tau=10}$')
ax.set_xlabel('$\\lambda$')
ax.set_xlim([-0.1,2.1])
ax.legend(loc='best')

# Inset
inset = fig.add_axes([0.2,0.22,0.33,0.33])
inset.scatter(lamdas, err0, color='black', s=8)
inset.scatter(lamdas, err1, color='red', s=8)
inset.scatter(lamdas, err2, color='blue', s=8)
inset.set_yscale('log')
inset.set_xlim([-0.1,2.1])
#inset.set_xlabel('$\\lambda$')
inset.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig('couplings.png')

# SPARSE
dat_cv_huge = read_data('data/huge-solved.dat')
raw_ok = dat_cv_huge[:,1]/dat_cv_huge[:,2] > 2
cv_ok = dat_cv_huge[:,3]/dat_cv_huge[:,4] > 2
dat_raw = dat_cv_huge[raw_ok]
dat_cv = dat_cv_huge[cv_ok]
plt.figure(figsize=(6,4), dpi=500)
plt.errorbar(dat_raw[:,0]-.15, dat_raw[:,1], yerr=dat_raw[:,2], fmt='.', label='No VR')
plt.errorbar(dat_cv[:,0]+.15, dat_cv[:,3], yerr=dat_cv[:,4], fmt='.', label='Sparse VR')
plt.yscale('log')
plt.ylabel('$\\langle\\phi(\\tau)\\phi(0)\\rangle$')
plt.xlabel('$\\tau$')
plt.xlim([-0.1,20.1])
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('sparse.png')

