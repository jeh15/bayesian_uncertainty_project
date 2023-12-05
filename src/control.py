import os
import argparse
import pickle
import time

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax
import numpy as np
import numpyro
from numpyro.diagnostics import hpdi, summary

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

    # Summary statistics:
    summary_dict = summary(samples, 0.95, False)
    layer_1_norm = np.sqrt(
        (summary_dict['w1']['97.5%'] - summary_dict['w1']['2.5%']) ** 2
    )
    layer_2_norm = np.sqrt(
        (summary_dict['w2']['97.5%'] - summary_dict['w2']['2.5%']) ** 2
    )
    layer_3_norm = np.sqrt(
        (summary_dict['w3']['97.5%'] - summary_dict['w3']['2.5%']) ** 2
    )

    layer_1_mean = summary_dict['w1']['mean']
    layer_2_mean = summary_dict['w2']['mean']
    layer_3_mean = summary_dict['w3']['mean']

    layer_1_std = summary_dict['w1']['std']
    layer_2_std = summary_dict['w2']['std']
    layer_3_std = summary_dict['w3']['std']

    environment_name = "mass_spring_damper"
    # environment_name = "cartpole"

    # Num Dead Neurons:
    layer_number = 1
    for layer_mean, layer_std in zip(
        [layer_1_mean, layer_2_mean, layer_3_mean], [layer_1_std, layer_2_std, layer_3_std]
    ):
        mean_mask = np.zeros_like(layer_mean)
        std_mask = np.zeros_like(layer_std)
        layer_zero_mean = np.where(np.abs(layer_mean) <= 1e-3)
        layer_zero_std = np.where(np.abs(layer_std) <= 1e-3)
        mean_mask[layer_zero_mean] = 1.0
        std_mask[layer_zero_std] = 1.0
        dead_neurons = np.where(mean_mask + std_mask == 2.0)[0].shape[0]
        print(f"Layer {layer_number} number of dead neurons: {dead_neurons}")
        layer_number += 1

    # Layer 1 Plot:
    fig, ax = plt.subplots(1)
    im = ax.imshow(layer_1_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('Layer 1 Distribution Range')
    ax.set_yticks(np.arange(layer_1_norm.shape[0]))
    fig.tight_layout()

    filename = f'{environment_name}_layer_1_CI.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    fig, ax = plt.subplots(1)
    im = ax.imshow(layer_2_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('Layer 2 Distribution Range')
    fig.tight_layout()

    filename = f'{environment_name}_layer_2_CI.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    fig, ax = plt.subplots(1)
    im = ax.imshow(layer_3_norm)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="50%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('Layer 3 Distribution Range')
    ax.set_yticks(np.arange(layer_3_norm.shape[0]))
    fig.tight_layout()

    filename = f'{environment_name}_layer_3_CI.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    # Layer Means:
    fig, ax = plt.subplots(1)
    im = ax.imshow(layer_1_mean)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('Layer 1 Mean')
    ax.set_yticks(np.arange(layer_1_mean.shape[0]))
    fig.tight_layout()

    filename = f'{environment_name}_layer_1_mean.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    fig, ax = plt.subplots(1)
    im = ax.imshow(layer_2_mean)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('Layer 2 Mean')
    fig.tight_layout()

    filename = f'{environment_name}_layer_2_mean.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    fig, ax = plt.subplots(1)
    im = ax.imshow(layer_3_mean)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="50%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('Layer 3 Mean')
    ax.set_yticks(np.arange(layer_3_mean.shape[0]))
    fig.tight_layout()

    filename = f'{environment_name}_layer_3_mean.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    # Layer Standard Deviations:
    fig, ax = plt.subplots(1)
    im = ax.imshow(layer_1_std)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('Layer 1 Standard Deviation')
    ax.set_yticks(np.arange(layer_1_std.shape[0]))
    fig.tight_layout()

    filename = f'{environment_name}_layer_1_std.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    fig, ax = plt.subplots(1)
    im = ax.imshow(layer_2_std)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('Layer 2 Standard Deviation')
    fig.tight_layout()

    filename = f'{environment_name}_layer_2_std.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    fig, ax = plt.subplots(1)
    im = ax.imshow(layer_3_std)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="50%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_title('Layer 3 Standard Deviation')
    ax.set_yticks(np.arange(layer_3_std.shape[0]))
    fig.tight_layout()

    filename = f'{environment_name}_layer_3_std.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

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
    if environment_name == "mass_spring_damper":
        Q = np.array([
            [1, 0],
            [0, 0.5]
        ])
        R = 0.1 * np.eye(1)
    else:
        Q = np.array([
            [0.01, 0, 0, 0],
            [0, 0.01, 0, 0],
            [0, 0, 1000.0, 0],
            [0, 0, 0, 10.0],
        ])
        R = 1.0 * np.eye(1)
        state = np.array([[0.0, 0.0, 0.01, 0.0]])

    state = np.concatenate([state, action], axis=1)
    states = [state.flatten()]
    predicted_states = [state.flatten()[:-1]]
    percentiles = []
    inference_time = []
    step_time = []
    start_time = time.time()

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
        if environment_name == "mass_spring_damper":
            B = np.array([[1.0], [0.0]])
        else:
            B = np.array([[1.0], [0.0], [0.0], [0.0]])

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
        percentiles.append(hpdi(predicted_state, 0.95))

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
    if environment_name == "mass_spring_damper":
        pass
    else:
        th_percentiles = percentiles[:, :, 2]
        dth_percentiles = percentiles[:, :, 3]
        generate_video(config, states[:, :-1], "cart_pole")

    # Create plots:
    x_range = np.arange(states[:, 0].shape[0]) * config.simulation_params.dt
    percentile_range = np.arange(1, x_percentiles.shape[0]+1) * config.simulation_params.dt
    if environment_name == "mass_spring_damper":
        fig, ax = plt.subplots(3, constrained_layout=True)
        ax[0].plot(x_range, states[:, 0])
        ax[0].plot(x_range, predicted_states[:, 0])
        ax[0].fill_between(
            percentile_range, x_percentiles[:, 0], x_percentiles[:, 1], color="lightblue"
        )
        ax[0].set(xlabel="t", ylabel="x")
        ax[0].set_ylim([-3.0, 3.0])
        ax[1].plot(x_range, states[:, 1])
        ax[1].plot(x_range, predicted_states[:, 1])
        ax[1].fill_between(
            percentile_range, dx_percentiles[:, 0], dx_percentiles[:, 1], color="lightblue"
        )
        ax[1].set(xlabel="t", ylabel="dx")
        ax[1].set_ylim([-3.0, 3.0])
        ax[2].plot(x_range, states[:, 2])
        ax[2].set(xlabel="t", ylabel="u")
        ax[2].set_ylim([-3.0, 3.0])
    else:
        fig, ax = plt.subplots(5, constrained_layout=True)
        ax[0].plot(x_range, states[:, 0])
        ax[0].plot(x_range, predicted_states[:, 0])
        ax[0].fill_between(
            percentile_range, x_percentiles[:, 0], x_percentiles[:, 1], color="lightblue"
        )
        ax[0].set(xlabel="t", ylabel="x")
        ax[0].set_ylim([-3.5, 3.5])

        ax[1].plot(x_range, states[:, 1])
        ax[1].plot(x_range, predicted_states[:, 1])
        ax[1].fill_between(
            percentile_range, dx_percentiles[:, 0], dx_percentiles[:, 1], color="lightblue"
        )
        ax[1].set(xlabel="t", ylabel="dx")
        ax[1].set_ylim([-3.5, 3.5])
        ax[2].plot(x_range, states[:, 2])
        ax[2].plot(x_range, predicted_states[:, 2])
        ax[2].fill_between(
            percentile_range, th_percentiles[:, 0], th_percentiles[:, 1], color="lightblue"
        )
        ax[2].set(xlabel="t", ylabel="theta")
        ax[2].set_ylim([-2*np.pi, 2*np.pi])
        ax[3].plot(x_range, states[:, 3])
        ax[3].plot(x_range, predicted_states[:, 3])
        ax[3].fill_between(
            percentile_range, dth_percentiles[:, 0], dth_percentiles[:, 1], color="lightblue"
        )
        ax[3].set(xlabel="t", ylabel="dtheta")
        ax[3].set_ylim([-5.0, 5.0])
        ax[4].plot(x_range, states[:, 4])
        ax[4].set(xlabel="t", ylabel="u")
        ax[4].set_ylim([-5.0, 5.0])

    filename = f"{environment_name}_credible_plot.pdf"
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
