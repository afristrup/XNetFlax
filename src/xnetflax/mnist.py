import jax.numpy as jnp
import numpy as np
import chex
from typing import Tuple
from datasets import load_dataset


def get_dataset_hf() -> Tuple[chex.Array, chex.Array]:
    """Load the MNIST dataset from Hugging Face.
    Source: https://flax-linen.readthedocs.io/en/latest/guides/data_preprocessing/loading_datasets.html"""
    mnist = load_dataset("mnist")

    ds = {}
    for split in ["train", "test"]:
        ds[split] = {
            "image": np.array([np.array(image) for image in mnist[split]["image"]]),
            "label": np.array(mnist[split]["label"]),
        }

        # Cast to jnp and rescale pixel values
        ds[split]["image"] = jnp.float32(ds[split]["image"] / 255.0)
        ds[split]["label"] = jnp.int16(ds[split]["label"])

        # Append trailing channel dimension
        ds[split]["image"] = jnp.expand_dims(ds[split]["image"], 3)

    return ds["train"], ds["test"]
