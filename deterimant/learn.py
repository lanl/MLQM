#!/usr/bin/env python

import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="Learn",
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


