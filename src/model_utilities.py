from typing import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro import deterministic


def model(
    X: jax.Array,
    Y: jax.Array,
    num_hidden: int,
    num_output: int,
) -> None:
    N, num_input = X.shape

    w1 = numpyro.sample(
        "w1",
        dist.Normal(
            jnp.zeros((num_input, num_hidden)), jnp.ones((num_input, num_hidden))
        ),
    )
    z1 = jnp.tanh(jnp.matmul(X, w1))

    w2 = numpyro.sample(
        "w2",
        dist.Normal(
            jnp.zeros((num_hidden, num_hidden)), jnp.ones((num_hidden, num_hidden))
        ),
    )
    z2 = jnp.tanh(jnp.matmul(z1, w2))

    w3 = numpyro.sample(
        "w3",
        dist.Normal(
            jnp.zeros((num_hidden, num_output)), jnp.ones((num_hidden, num_output))
        ),
    )
    z3 = jnp.matmul(z2, w3)

    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=Y)


def stein_model(
    x: jax.Array,
    y: jax.Array,
    num_hidden: int,
    num_output: int,
    subsample_size: int = 100,
):
    """BNN described in section 5 of [1].

    **References:**
        1. *Stein variational gradient descent: A general purpose bayesian inference algorithm*
        Qiang Liu and Dilin Wang (2016).
    """

    prec_nn = numpyro.sample(
        "prec_nn", dist.Gamma(1.0, 0.1)
    )  # hyper prior for precision of nn weights and biases

    n, num_input = x.shape

    with numpyro.plate("l1_hidden", num_hidden, dim=-1):
        # prior l1 bias term
        b1 = numpyro.sample(
            "nn_b1",
            dist.Normal(
                jnp.zeros((num_hidden,)), 1.0 / jnp.sqrt(prec_nn) * jnp.ones((num_hidden,)),
            ),
        )

        with numpyro.plate("l1_feat", num_input, dim=-2):
            w1 = numpyro.sample(
                "nn_w1",
                dist.Normal(
                    jnp.zeros((num_input, num_hidden)), 1.0 / jnp.sqrt(prec_nn) * jnp.ones((num_input, num_hidden)),
                )
            )  # prior on l1 weights

    with numpyro.plate("l2_hidden", num_output, dim=-1):
        w2 = numpyro.sample(
            "nn_w2",
            dist.Normal(
                jnp.zeros((num_hidden, num_output)), 1.0 / jnp.sqrt(prec_nn) * jnp.ones((num_hidden, num_output)),
            )
        )  # prior on output weights

    b2 = numpyro.sample(
        "nn_b2",
        dist.Normal(
            jnp.zeros((num_output,)), 1.0 / jnp.sqrt(prec_nn) * jnp.ones((num_output,)),
        )
    )  # prior on output bias term

    # precision prior on observations
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(1.0, 0.1))
    with numpyro.plate(
        "data",
        x.shape[0],
        subsample_size=subsample_size,
    ):
        batch_x = numpyro.subsample(x, event_dim=1)
        if y is not None:
            batch_y = numpyro.subsample(y, event_dim=1)
        else:
            batch_y = y

        loc_y = deterministic(
            "y_pred", jnp.maximum(batch_x @ w1 + b1, 0) @ w2 + b2,
        )

        numpyro.sample(
            "y",
            dist.Normal(
                loc_y, 1.0 / jnp.sqrt(prec_obs) * jnp.ones((subsample_size, num_output))
            ).to_event(1),  # 1 hidden layer with ReLU activation
            obs=batch_y,
        )


def train(
    model: Callable,
    args: NamedTuple,
    rng_key: PRNGKey,
    X: jax.Array,
    Y: jax.Array,
) -> MCMC:
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=True,
    )
    mcmc.run(rng_key, X, Y, args.num_hidden, args.num_output)
    return mcmc


def run_inference(
    model: Callable,
    num_hidden: int,
    num_output: int,
    samples: dict[str, jax.Array],
    rng_key: PRNGKey,
    X: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    prediction_fn = Predictive(model, samples)
    jacobian_fn = jax.jacfwd(Predictive(model, samples), argnums=1)
    model_prediction = jnp.squeeze(
        prediction_fn(rng_key, X, None, num_hidden, num_output)["Y"]
    )
    model_jacobian = jnp.squeeze(
        jacobian_fn(rng_key, X, None, num_hidden, num_output)["Y"]
    )
    return model_prediction, model_jacobian
