import cv2
import jax
import json
import pickle
import haiku as hk
import tensorflow as tf
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

from pathlib import Path
from einops import rearrange
from argparse import ArgumentParser
from tempfile import TemporaryDirectory
from mlflow.tracking import MlflowClient

from models import VisionTransformer


def parse_arguments():
    parser = ArgumentParser("Load Vision Transformer and perform inference.")
    parser.add_argument("--experiment", type=str, default="cifar_100", help="Experiment name to extract")
    parser.add_argument("--run", type=str, help="Run to load")
    parser.add_argument("--image", type=str, help="Path to image")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    client = MlflowClient()
    with TemporaryDirectory() as temp_dir:
        client.download_artifacts(args.run, "config.json", temp_dir)
        temp_file = Path(temp_dir, "config.json")
        with open(temp_file, "r") as f:
            config = json.load(f)

    def create_transformer(x):
        return VisionTransformer(
            config["k"],
            config["heads"],
            config["depth"],
            config["num_classes"],
            config["patch_size"],
            config["image_size"],
            config["dropout"],
        )(x)

    transformer = hk.transform(create_transformer)

    with open("weights.pkl", "rb") as f:
        params = pickle.load(f)

    key = random.PRNGKey(0)
    image = cv2.imread(args.image)
    image = cv2.resize(image, config["image_size"])
    image = jnp.array(rearrange(image, "h w c -> 1 h w c"))
    image = tf.image.per_image_standardization(image).numpy()
    _, rollout = transformer.apply(params, key, image)
    rollout_resized = jax.image.resize(rollout[0], config["image_size"], method="linear")

    plt.imshow(image[0])
    plt.imshow(rollout_resized, cmap="jet", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
