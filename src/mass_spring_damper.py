import functools
from typing import Self

from ml_collections import config_dict
import numpy as np
import jax
import jax.numpy as jnp

# type aliases:
from jax.random import PRNGKey


class MassSpringDamper():
    def __init__(
        self,
        config: config_dict.ConfigDict,
    ) -> None:
        # Store Config:
        self.config = config
        self.model_params = config.model_params
        self.simulation_params = config.simulation_params
        # Calculate Derived Parameters:
        self.model_params.omega = np.sqrt(self.model_params.k / self.model_params.m)
        self.model_params.zeta = self.model_params.c / (2 * self.model_params.m * self.model_params.omega)

    @functools.partial(jax.jit, static_argnames=("self"))
    def step(
        self: Self,
        state: jax.typing.ArrayLike,
        action: jax.typing.ArrayLike,
    ) -> jnp.ndarray:
        ddx = (
            action / self.model_params.m
            - self.model_params.omega**2 * state[0]
            - 2 * self.model_params.zeta * self.model_params.omega * state[1]
        )
        dx = state[1] + ddx * self.simulation_params.dt
        x = state[0] + dx * self.simulation_params.dt
        return jnp.array([x, dx])

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
                state + jax.random.normal(
                    key=rng_key,
                    shape=(2,),
                ) * self.simulation_params.noise_std,
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
        states = jax.random.normal(
            key=rng_key,
            shape=(num_batches, 2),
        )
        actions = jax.random.uniform(
            key=rng_key,
            shape=(num_batches, 1),
            minval=-1.0,
            maxval=1.0,
        )
        return states, actions


def create_configuration() -> config_dict.ConfigDict:
    config = config_dict.ConfigDict()
    config.model_params = config_dict.ConfigDict()
    config.simulation_params = config_dict.ConfigDict()
    config.model_params.c = 1.0
    config.model_params.m = 1.0
    config.model_params.k = 1.0
    config.simulation_params.dt = 0.01
    config.simulation_params.noise_std = 0.001
    return config


if __name__ == "__main__":
    config = create_configuration()
    environment = MassSpringDamper(config)
    carry, data = environment.generate_data(
        num_batches=100,
        num_steps=100,
        rng_key=jax.random.PRNGKey(0),
        add_noise=True,
    )
