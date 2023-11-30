import os
import argparse
import pickle
import time

import matplotlib.pyplot as plt
import jax
import numpy as np
import numpyro
import scipy

from model_utilities import model, run_inference
from control_utilities import LQR

from mass_spring_damper import MassSpringDamper as environment
# from cartpole import CartPole as environment

jax.config.update("jax_enable_x64", True)


def main(args):
    # Set initial RNG Key:
    initial_key = jax.random.PRNGKey(args.seed)
    state_key, predict_key = jax.random.split(initial_key)

    # Load trained model:
    filename = args.model_path
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    with open(filepath, 'rb') as handle:
        mcmc, config, train_args = pickle.load(handle)

    env = environment(config)
    state, action = env.random_states(
        num_batches=1,
        rng_key=state_key,
    )

    samples = mcmc.get_samples()

    input_layer = samples['w1']
    hidden_layer = samples['w2']
    output_layer = samples['w3']

    # Plot weight distributions:
    layers = [input_layer, hidden_layer, output_layer]
    # for layer in zip(layers):
    #     layer = layer[0]
    #     rows, cols = layer.shape[1], layer.shape[2]
    #     fig, ax = plt.subplots(rows, cols, constrained_layout=True)
    #     for i in range(rows):
    #         for j in range(cols):
    #             layer_kernel = scipy.stats.gaussian_kde(layer[:, i, j])
    #             layer_range = layer_kernel.dataset.min(), layer_kernel.dataset.max()
    #             layer_range = np.linspace(*layer_range, 1000)
    #             pdf = layer_kernel.pdf(layer_range)
    #             ax[i, j].plot(layer_range, pdf)
    #     plt.show()

    # Plot average weight sample:
    layers = [input_layer, hidden_layer, output_layer]
    for layer in zip(layers):
        layer = layer[0]
        rows, cols = layer.shape[1], layer.shape[2]
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
        layer_average = np.mean(layer, axis=0)
        for i in range(rows):
            for j in range(cols):
                text = ax.text(
                    j,
                    i,
                    np.format_float_positional(layer_average[i, j], precision=1),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=8,
                )
        ax.imshow(layer_average)
        plt.show()

    filename = "mass_spring_damper_lqr.pdf"
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    plt.savefig(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained model with LQR controller.")
    parser.add_argument("--seed", nargs="?", default=0, type=int)
    parser.add_argument("--model-path", nargs="?", default="mass_spring_damper.pickle", type=str, help='path to trained model.')
    parser.add_argument("--num-devices", nargs="?", default=1, type=int, help='number of devices')
    parser.add_argument("--device", default="gpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.enable_x64(use_x64=True)
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_devices)

    main(args)
