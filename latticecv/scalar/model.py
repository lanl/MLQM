from dataclasses import dataclass

import jax
import jax.numpy as jnp

@dataclass
class Model:
    L: int
    m2: float
    lamda: float

    @staticmethod
    def from_file(filename):
        v = {}
        with open(filename) as f:
            exec(f.read(), None, v)
        L = v['L']
        m2 = v['m2']
        lamda = v['lamda']
        return Model(L, m2, lamda)

    def field_near(self, phi, x, y):
        phi = phi.reshape((self.L,self.L))
        xm, xp = (x-1)%self.L, (x+1)%self.L
        ym, yp = (y-1)%self.L, (y+1)%self.L
        return phi[xm,y], phi[x,ym], phi[x,y], phi[xp,y], phi[x,yp]

    def action_local(self, phimx, phimy, phi, phipx, phipy):
        pot = self.m2/2. * jnp.abs(phi)**2 + self.lamda/24. * jnp.abs(phi)**4
        kinx = ((phipx-phi)**2 + (phimx-phi)**2) / 2.
        kiny = ((phipy-phi)**2 + (phimy-phi)**2) / 2.
        return kinx + kiny + pot

    def action(self, phi):
        phi = phi.reshape((self.L,self.L))
        phix = jnp.roll(phi, 1, axis=0)
        phiy = jnp.roll(phi, 1, axis=1)
        kinx = jnp.sum((phi-phix)**2)/2.
        kiny = jnp.sum((phi-phiy)**2)/2.
        pot = jnp.sum(self.m2/2. * phi**2 + self.lamda/24. * phi**4)
        return kinx + kiny + pot

