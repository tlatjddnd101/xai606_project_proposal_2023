from typing import Callable, Sequence
from flax import linen as nn
import jax.numpy as jnp


def default_init():
    return nn.initializers.he_normal()


class BaselineClassifier(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        for i, size in enumerate(self.hidden_dims):
            if i+1 < len(self.hidden_dims):
                x = nn.Dense(size, kernel_init=default_init())(x)
                x = self.activations(x)
            elif i+1 == len(self.hidden_dims):
                x = nn.Dense(size, use_bias=False, kernel_init=default_init())(x)

        return x