#!/usr/bin/env python

import numpy as np

def bootstrap(xs, ws=None, N=300, Bs=50):
    if Bs > len(xs):
        Bs = len(xs)
    B = len(xs)//Bs
    if ws is None:
        ws = xs*0 + 1
    # Block
    x, w = [], []
    for i in range(Bs):
        x.append(sum(xs[i*B:i*B+B]*ws[i*B:i*B+B])/sum(ws[i*B:i*B+B]))
        w.append(sum(ws[i*B:i*B+B]))
    x = np.array(x)
    w = np.array(w)
    # Regular bootstrap
    y = x * w
    m = (sum(y) / sum(w))
    ms = []
    for n in range(N):
        s = np.random.choice(range(len(x)), len(x))
        ms.append((sum(y[s]) / sum(w[s])))
    ms = np.array(ms)
    return m, np.std(ms.real) + 1j*np.std(ms.imag)

a = 2.
x = a*np.random.normal(size=1000000)
if False:
    obs = x
    sub = 2*np.exp(-x**2/2.) * x
if True:
    obs = x**2
    #sub = np.exp(-x**2/(2.*a**2)) * (4*x**2/a**2 - 2.)
    sub = (x**2/a**2 - 1.)

print(bootstrap(sub))
for a in np.linspace(-1,4,51):
    print(a,np.std(obs-a*sub))
    #print(a, *bootstrap(obs-a*sub))

