import os
import argparse
import pickle
import time

import matplotlib.pyplot as plt
import jax
import numpy as np
import numpyro

from model_utilities import model, run_inference
from control_utilities import LQR

from mass_spring_damper import MassSpringDamper as environment
# from cartpole import generate_video, CartPole as environment

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

    inference_fn = lambda rng_key, X: run_inference(
        model,
        train_args.num_hidden,
        train_args.num_output,
        samples,
        rng_key,
        X,
    )
    jit_inference_fn = jax.jit(inference_fn)

    # LQR controller:
    # Mass Spring Damper:
    Q = np.array([
        [1, 0],
        [0, 0.5]
    ])
    R = 0.1 * np.eye(1)

    # Cartpole:
    # Q = np.array([
    #     [0.01, 0, 0, 0],
    #     [0, 0.01, 0, 0],
    #     [0, 0, 1000.0, 0],
    #     [0, 0, 0, 10.0],
    # ])
    # R = 1.0 * np.eye(1)
    # state = np.array([[0.0, 0.0, 0.01, 0.0]])

    state = np.concatenate([state, action], axis=1)
    states = [state.flatten()]
    predicted_states = [state.flatten()[:-1]]
    percentiles = []
    inference_time = []
    step_time = []
    start_time = time.time()

    # Runs slower than lqr.py?
    # Push to jax control flow? (Would have to make a jax LQR solver)
    num_steps = 1000
    for _ in range(num_steps):
        # Predict next state:
        start_inference_time = time.time()
        predicted_state, A = jit_inference_fn(predict_key, state)
        mean_predicted_state = np.mean(predicted_state, axis=0).flatten()

        inference_time.append(time.time() - start_inference_time)

        _, predict_key = jax.random.split(predict_key)
        # Average state matrix:
        A = np.mean(A, axis=0)
        # Isolate state and control matrices:
        # Old:
        # B = np.expand_dims(A[:, -1], axis=-1)
        # New:
        B = np.array([[1.0], [0.0]])
        # B = np.array([[1.0], [0.0], [0.0], [0.0]])

        A = A[:, :-1]
        # Compute control input:
        x = state[:, :-1].T
        u = LQR(A, B, Q, R, x)

        # Saturate control input:
        u = np.clip(u, -3.0, 3.0)
        # u = np.array([[0.0]])

        # Step forward:
        x = x.flatten()
        u = u.flatten()

        start_step_time = time.time()
        x = env.step(
            x,
            u,
        )
        step_time.append(time.time() - start_step_time)

        state = np.expand_dims(
            np.append(x, u),
            axis=0,
        )
        states.append(state.flatten())
        predicted_states.append(mean_predicted_state)
        percentiles.append(np.percentile(predicted_state, [5.0, 95.0], axis=0))

    end_time = time.time()
    print(f"Simulation Time: {config.simulation_params.dt * num_steps}")
    print(f"Total Time elapsed: {end_time - start_time}")
    print(f"Average Inference Time: {np.mean(inference_time)}")
    print(f"Average Step Time: {np.mean(step_time)}")

    # Numpy arrays:
    states = np.asarray(states)
    predicted_states = np.asarray(predicted_states)
    percentiles = np.asarray(percentiles)
    x_percentiles = percentiles[:, :, 0]
    dx_percentiles = percentiles[:, :, 1]

    # Cartpole Video:
    # generate_video(config, states[:, :-1], "cart_pole")

    # # Create plots:
    fig, ax = plt.subplots(3, constrained_layout=True)
    ax[0].plot(np.arange(states[:, 0].shape[0]), states[:, 0])
    ax[0].plot(np.arange(predicted_states[:, 0].shape[0]), predicted_states[:, 0])
    ax[0].fill_between(
        np.arange(1, x_percentiles.shape[0]+1), x_percentiles[:, 0], x_percentiles[:, 1], color="lightblue"
    )
    ax[0].set(xlabel="t", ylabel="x")
    ax[0].set_ylim([-3.0, 3.0])
    ax[1].plot(np.arange(states[:, 0].shape[0]), states[:, 1])
    ax[1].plot(np.arange(predicted_states[:, 0].shape[0]), predicted_states[:, 1])
    ax[1].fill_between(
        np.arange(1, dx_percentiles.shape[0]+1), dx_percentiles[:, 0], dx_percentiles[:, 1], color="lightblue"
    )
    ax[1].set(xlabel="t", ylabel="dx")
    ax[1].set_ylim([-3.0, 3.0])
    ax[2].plot(np.arange(states[:, 0].shape[0]), states[:, 2])
    ax[2].set(xlabel="t", ylabel="u")
    ax[2].set_ylim([-3.0, 3.0])

    # Create plots:
    # fig, ax = plt.subplots(5, constrained_layout=True)
    # ax[0].plot(np.arange(states[:, 0].shape[0]), states[:, 0])
    # ax[0].plot(np.arange(predicted_states[:, 0].shape[0]), predicted_states[:, 0])
    # ax[0].set(xlabel="t", ylabel="x")
    # ax[0].set_ylim([-3.5, 3.5])
    # ax[1].plot(np.arange(states[:, 0].shape[0]), states[:, 1])
    # ax[1].plot(np.arange(predicted_states[:, 0].shape[0]), predicted_states[:, 1])
    # ax[1].set(xlabel="t", ylabel="dx")
    # ax[1].set_ylim([-3.5, 3.5])
    # ax[2].plot(np.arange(states[:, 0].shape[0]), states[:, 2])
    # ax[2].plot(np.arange(predicted_states[:, 0].shape[0]), predicted_states[:, 2])
    # ax[2].set(xlabel="t", ylabel="theta")
    # ax[2].set_ylim([-2*np.pi, 2*np.pi])
    # ax[3].plot(np.arange(states[:, 0].shape[0]), states[:, 3])
    # ax[3].plot(np.arange(predicted_states[:, 0].shape[0]), predicted_states[:, 3])
    # ax[3].set(xlabel="t", ylabel="dtheta")
    # ax[3].set_ylim([-5.0, 5.0])
    # ax[4].plot(np.arange(states[:, 0].shape[0]), states[:, 4])
    # ax[4].set(xlabel="t", ylabel="u")
    # ax[4].set_ylim([-5.0, 5.0])

    filename = "msd_lqr_2.pdf"
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    plt.savefig(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained model with LQR controller.")
    parser.add_argument("--seed", nargs="?", default=0, type=int)
    parser.add_argument("--model-path", nargs="?", default="mass_spring_damper_reduced_v2.pickle", type=str, help='path to trained model.')
    parser.add_argument("--num-devices", nargs="?", default=1, type=int, help='number of devices')
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.enable_x64(use_x64=True)
    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_devices)

    main(args)
