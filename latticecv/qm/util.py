import jax
import jax.numpy as jnp

import numpy as np

def bootstrap_np(xs, ws=None, N=1000, Bs=50):
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

@jax.jit
def bootstrap(xs, N=100, Bs=50):
    if Bs > len(xs):
        Bs = len(xs)
    B = len(xs)//Bs
    # Block
    def bmean(i):
        #return jnp.mean(xs[i*B:i*B+B])
        return jnp.mean(jax.lax.dynamic_slice(xs,(i*B,),(B,)))
    x = jax.vmap(bmean)(jnp.arange(Bs))

    # Bootstrap
    def mean(k):
        s = jax.random.choice(k, Bs, shape=(Bs,))
        return jnp.mean(x[s])
    ms = jax.vmap(mean)(jax.random.split(jax.random.PRNGKey(0),N))
    return jnp.mean(x), jnp.std(ms) + 1j*jnp.std(ms)

