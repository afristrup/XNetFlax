from typing import Any, Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
import chex
from dataclasses import field
from flax.core import FrozenDict


class CauchyActivation(nn.Module):
    """Cauchy activation function with learnable parameters.

    Attributes:
        lambda1_init: Initial value for λ1 parameter
        lambda2_init: Initial value for λ2 parameter
        d_init: Initial value for d parameter
    """

    lambda1_init: float = 1.0
    lambda2_init: float = 1.0
    d_init: float = 1.0

    def setup(self):
        """Initialize the learnable parameters."""
        self.lambda1 = self.param(
            "lambda1", nn.initializers.constant(self.lambda1_init), ()
        )
        self.lambda2 = self.param(
            "lambda2", nn.initializers.constant(self.lambda2_init), ()
        )
        self.d = self.param("d", nn.initializers.constant(self.d_init), ())

    def __call__(self, x: chex.Array) -> chex.Array:
        """Apply the Cauchy activation function.

        Args:
            x: Input array of any shape.

        Returns:
            Array of same shape as input with Cauchy activation applied.
        """
        # Type and shape assertions
        chex.assert_type(x, float)
        chex.assert_equal_shape((x,))

        # Compute denominator term
        x2_d2 = jnp.square(x) + jnp.square(self.d)

        # Apply Cauchy activation
        return self.lambda1 * x / x2_d2 + self.lambda2 / x2_d2


# Example usage and testing
def test_cauchy_activation():
    """Test the CauchyActivation module."""
    # Initialize the module
    module = CauchyActivation()

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    variables = module.init(key, jnp.array([1.0]))

    # Test with sample input
    x = jnp.array([1.0, 2.0, 3.0])
    output = module.apply(variables, x)

    # Assertions
    chex.assert_shape(output, x.shape)
    chex.assert_type(output, float)

    return {"input": x, "output": output, "params": variables}


if __name__ == "__main__":
    results = test_cauchy_activation()
    print("Test passed successfully!")
    print(f"Sample input: {results['input']}")
    print(f"Sample output: {results['output']}")
    print(f"Parameters: {results['params']}")
