import os
import functools
from typing import Self, List

from ml_collections import config_dict
import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle
from tqdm import tqdm

# type aliases:
from jax.random import PRNGKey


class CartPole:
    def __init__(
        self,
        config: config_dict.ConfigDict,
    ) -> None:
        # Store Config:
        self.config = config
        self.model_params = config.model_params
        self.simulation_params = config.simulation_params

    @functools.partial(jax.jit, static_argnames=("self"))
    def step(
        self: Self,
        state: jax.Array,
        action: jax.Array,
    ) -> jnp.ndarray:
        def ddx_fn(
            state: jax.Array,
            action: jax.Array,
        ) -> jnp.ndarray:
            th = state[2]
            dth = state[3]
            u = action[0]

            # Equation of Motion for ddx:
            a = u / (
                self.model_params.mass_cart
                + self.model_params.mass_pole * jnp.sin(th) ** 2
            )
            b = (
                self.model_params.mass_pole
                * self.model_params.length
                * dth**2
                * jnp.sin(th)
                / (
                    self.model_params.mass_cart
                    + self.model_params.mass_pole * jnp.sin(th) ** 2
                )
            )
            c = (
                self.model_params.mass_pole
                * self.simulation_params.gravity
                * jnp.sin(th)
                * jnp.cos(th)
                / (
                    self.model_params.mass_cart
                    + self.model_params.mass_pole * jnp.sin(th) ** 2
                )
            )
            return a + b + c

        def ddth_fn(
            state: jax.Array,
            action: jax.Array,
        ) -> jnp.ndarray:
            th = state[2]
            dth = state[3]
            u = action[0]

            # Equation of Motion for ddth:
            a = (
                -u
                * jnp.cos(th)
                / (
                    self.model_params.mass_cart * self.model_params.length
                    + self.model_params.mass_pole
                    * self.model_params.length
                    * jnp.sin(th) ** 2
                )
            )
            b = (
                -self.model_params.mass_pole
                * self.model_params.length
                * dth**2
                * jnp.sin(th)
                * jnp.cos(th)
                / (
                    self.model_params.mass_cart * self.model_params.length
                    + self.model_params.mass_pole
                    * self.model_params.length
                    * jnp.sin(th) ** 2
                )
            )
            c = (
                -(self.model_params.mass_cart + self.model_params.mass_pole)
                * self.simulation_params.gravity
                * jnp.sin(th)
                / (
                    self.model_params.mass_cart * self.model_params.length
                    + self.model_params.mass_pole
                    * self.model_params.length
                    * jnp.sin(th) ** 2
                )
            )
            return a + b + c

        # Euler Integration:
        ddx = ddx_fn(state, action)
        dx = state[1] + ddx * self.simulation_params.dt
        x = state[0] + dx * self.simulation_params.dt
        ddth = ddth_fn(state, action)
        dth = state[3] + ddth * self.simulation_params.dt
        th = state[2] + dth * self.simulation_params.dt

        return jnp.array([x, dx, th, dth])

    def generate_data(
        self: Self,
        num_batches: int,
        num_steps: int = 100,
        rng_key: PRNGKey = jax.random.PRNGKey(0),
        add_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        def _scan_fn(
            carry: tuple[jnp.ndarray, jnp.ndarray, PRNGKey],
            data: jnp.ndarray,
        ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, PRNGKey], jnp.ndarray]:
            state, action, rng_key = carry
            state = self.step(state, action).flatten()
            state = jnp.where(
                add_noise,
                state
                + jax.random.normal(
                    key=rng_key,
                    shape=(4,),
                )
                * self.simulation_params.noise_std,
                state,
            )
            _, rng_key = jax.random.split(rng_key)
            return (state, action, rng_key), state

        states, actions = self.random_states(num_batches, rng_key)

        scan_fn = lambda states, actions: jax.lax.scan(
            _scan_fn,
            (states, actions, rng_key),
            None,
            num_steps,
        )

        generate_fn = jax.vmap(
            scan_fn,
            in_axes=(0),
            out_axes=(0, 0),
        )

        carry, data = generate_fn(states, actions)
        initial_input = np.concatenate([states, actions], axis=1)

        return initial_input, np.squeeze(data)

    def random_states(
        self: Self,
        num_batches: int,
        rng_key: PRNGKey = jax.random.PRNGKey(0),
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # x_states = jax.random.normal(
        #     key=rng_key,
        #     shape=(num_batches, 2),
        # )
        x_states = jax.random.uniform(
            key=rng_key,
            shape=(num_batches, 1),
            minval=-2.0,
            maxval=2.0,
        )
        dx_states = jax.random.uniform(
            key=rng_key,
            shape=(num_batches, 1),
            minval=-1.0,
            maxval=1.0,
        )
        th_states = jax.random.uniform(
            key=rng_key,
            shape=(num_batches, 1),
            minval=-20 * jnp.pi / 180,
            maxval=20 * jnp.pi / 180,
        )
        dth_states = jax.random.uniform(
            key=rng_key,
            shape=(num_batches, 1),
            minval=-20 * jnp.pi / 180,
            maxval=20 * jnp.pi / 180,
        )
        actions = jax.random.uniform(
            key=rng_key,
            shape=(num_batches, 1),
            minval=-5.0,
            maxval=5.0,
        )

        # states = jnp.concatenate([x_states, th_states], axis=1)
        states = jnp.concatenate([x_states, dx_states, th_states, dth_states], axis=1)
        return states, actions


def generate_video(
        config: config_dict.ConfigDict,
        states: npt.ArrayLike,
        name: str,
):
    # Create plot handles for visualization:
    fig, ax = plt.subplots()
    pole, = ax.plot([], [], color='royalblue', zorder=10)
    lb, ub = -2.4, 2.4
    ax.axis('equal')
    ax.set_xlim([lb, ub])
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Cart Pole Simulation:')

    # Initialize Patch: (Cart)
    width = config.visualization_params.cart_width
    height = config.visualization_params.cart_height
    radius = config.visualization_params.radius
    xy_cart = (0, 0)
    cart_patch = Rectangle(
        xy_cart, width, height, color='cornflowerblue', zorder=5,
    )
    mass_patch = Circle(
        (0, 0), radius=radius, color='cornflowerblue', zorder=15,
    )
    ax.add_patch(cart_patch)
    ax.add_patch(mass_patch)

    # Ground:
    ground = ax.hlines(0, lb, ub, colors='black', linestyles='--', zorder=0)

    # Create video writer:
    fps = 24
    rate = int(1.0 / (config.simulation_params.dt * fps))
    writer_obj = FFMpegWriter(fps=fps)
    video_length = states.shape[0]
    filename = f"{name}.mp4"
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    with writer_obj.saving(fig, filepath, 300):
        for simulation_step in tqdm(range(0, video_length, rate)):
            fig, writer_obj, (cart_patch, mass_patch) = _visualize(
                fig=fig,
                writer_obj=writer_obj,
                plt=pole,
                patch=(cart_patch, mass_patch),
                state=states[simulation_step],
                config=config,
            )


def _visualize(
    fig,
    writer_obj,
    plt,
    patch,
    state,
    config,
):
    # Cart Position:
    cart_x = state[0]
    cart_y = 0.0

    # Pole Position:
    pole_x = -config.model_params.length * np.sin(state[2]) + cart_x
    pole_y = config.model_params.length * np.cos(state[2]) + cart_y

    # Update Pole:
    plt.set_data(
        [cart_x, pole_x],
        [cart_y, pole_y],
    )
    patch[1].center = pole_x, pole_y

    # Update Cart:
    patch[0].set(
        xy=(
            cart_x - config.visualization_params.cart_width / 2,
            cart_y - config.visualization_params.cart_height / 2,
        ),
    )

    # Update Drawing:
    fig.canvas.draw()
    # Grab and Save Frame:
    writer_obj.grab_frame()
    return fig, writer_obj, patch


def create_configuration() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()
    config.model_params = config_dict.ConfigDict()
    config.simulation_params = config_dict.ConfigDict()
    config.visualization_params = config_dict.ConfigDict()
    # Model:
    config.model_params.mass_cart = 1.0
    config.model_params.mass_pole = 1.0
    config.model_params.length = 0.2
    # Simulation:
    config.simulation_params.gravity = -9.81
    config.simulation_params.dt = 0.001
    config.simulation_params.noise_std = 0.001
    # Visualization:
    config.visualization_params.cart_width = 0.2
    config.visualization_params.cart_height = 0.1
    config.visualization_params.radius = 0.01
    return config


if __name__ == "__main__":
    config = create_configuration()
    environment = CartPole(config)
    carry, data = environment.generate_data(
        num_batches=1,
        num_steps=1000,
        rng_key=jax.random.PRNGKey(0),
        add_noise=False,
    )
    generate_video(config, data, "cart_pole")
