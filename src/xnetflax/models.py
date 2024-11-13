import jax
import jax.numpy as jnp

from flax import linen as nn

from xnetflax.activation import CauchyActivation


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


class CauchyCNN(nn.Module):
    """A simple CNN model with Cauchy activations."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = CauchyActivation()(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = CauchyActivation()(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = CauchyActivation()(x)
        x = nn.Dense(features=10)(x)
        return x


def test_cnn() -> None:
    model = CNN()
    x = jnp.ones((1, 28, 28, 1))
    y = model.apply(model.init(jax.random.PRNGKey(0), x), x)
    assert y.shape == (1, 10)
    print("CNN test successful!")


def test_cauchy_cnn() -> None:
    model = CauchyCNN()
    x = jnp.ones((1, 28, 28, 1))
    y = model.apply(model.init(jax.random.PRNGKey(0), x), x)
    assert y.shape == (1, 10)
    print("CauchyCNN test successful!")


if __name__ == "__main__":
    test_cnn()
    test_cauchy_cnn()
