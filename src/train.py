import os
import argparse
import pickle

import jax
import numpyro

# Mass Spring Damper:
from mass_spring_damper import MassSpringDamper as environment
from mass_spring_damper import create_configuration

# Cartpole:
# from cartpole import CartPole as environment
# from cartpole import create_configuration

import model_utilities

jax.config.update("jax_enable_x64", True)


def main(args):
    # Set initial RNG Key:
    rng_key = jax.random.PRNGKey(args.seed)

    # Parse arguments:
    num_batches = args.num_data
    num_steps = 1

    # Load configuration:
    config = create_configuration()

    # Initialize env:
    env = environment(config)
    X, Y = env.generate_data(
        num_batches=num_batches,
        num_steps=num_steps,
        rng_key=rng_key,
        add_noise=True,
    )

    # Add model params to args namespace:
    args.num_output = Y.shape[-1]

    # Run inference:
    _, rng_key = jax.random.split(rng_key)
    mcmc = model_utilities.train(
        model_utilities.model,
        args,
        rng_key,
        X,
        Y,
    )

    # Save trained model and configuration:
    pickle_data = [mcmc, config, args]
    filepath = os.path.join(
        os.path.dirname(__file__),
        args.filename,
    )
    with open(filepath, 'wb') as handle:
        pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian neural network: training script")
    parser.add_argument("--seed", nargs="?", default=0, type=int)
    parser.add_argument("--filename", nargs="?", default="mass_spring_damper.pickle", type=str)
    parser.add_argument("-n", "--num-samples", nargs="?", default=250, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=250, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-data", nargs="?", default=100000, type=int)
    parser.add_argument("--num-hidden", nargs="?", default=8, type=int)
    parser.add_argument("--device", default="gpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)

    main(args)
