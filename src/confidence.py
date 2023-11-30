import os
import argparse
import pickle

import matplotlib.pyplot as plt
import jax
import numpy as np
import numpyro
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

jax.config.update("jax_enable_x64", True)


def main(args):
    # Set initial RNG Key:
    initial_key = jax.random.PRNGKey(args.seed)

    # Load trained model:
    filename = args.model_path
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    with open(filepath, 'rb') as handle:
        mcmc, config, train_args = pickle.load(handle)

    samples = mcmc.get_samples()