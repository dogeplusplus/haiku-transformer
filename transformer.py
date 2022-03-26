import jax
import optax
import pickle
import mlflow
import typing as t
import haiku as hk
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


from tqdm import tqdm
from argparse import ArgumentParser
from collections import defaultdict

from models import VisionTransformer


def resize_image(example):
    image = tf.image.resize(example["image"], [72, 72])
    label = example["label"]
    image = tf.image.per_image_standardization(image)

    return image, label


def update_metrics(step, metrics, new):
    for name, value in new.items():
        metrics[name] = (metrics[name] * (step - 1) + value) / step

    return metrics


def parse_arguments():
    parser = ArgumentParser("Train Vision Transformer")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int,
                        help="Batch size", default=256)
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs", default=10)
    parser.add_argument(
        "--k", type=int, help="Dimension of transformer blocks", default=64)
    parser.add_argument("--heads", type=int, help="Number of heads", default=4)
    parser.add_argument("--patch-size", type=int,
                        help="Patch size to cut the image into.", default=6)
    parser.add_argument("--dropout", type=int,
                        help="Dropout probability", default=0.1)
    parser.add_argument("--depth", type=int,
                        help="Number of transformer blocks", default=2)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    tf.config.set_visible_devices([], 'GPU')

    num_classes = 100
    save_every = 10
    show_every = 5
    image_size = (72, 72)

    def create_transformer(x):
        return VisionTransformer(
            args.k,
            args.heads,
            args.depth,
            num_classes,
            args.patch_size,
            image_size,
            args.dropout,
        )(x)

    dataset_name = "cifar100"
    train_ds, val_ds = tfds.load(
        dataset_name,
        split=["train", "test"],
        shuffle_files=True,
    )
    train_ds = train_ds.map(resize_image).batch(
        args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(resize_image).batch(
        args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = tfds.as_numpy(train_ds)
    val_ds = tfds.as_numpy(val_ds)

    transformer = hk.transform(create_transformer)
    xs, _ = next(iter(train_ds))

    rng_seq = hk.PRNGSequence(42)
    params = transformer.init(next(rng_seq), xs)
    param_count = sum(x.size for x in jax.tree_leaves(params))
    decay_steps = 10000
    lr_scheduler = optax.cosine_decay_schedule(1e-3, decay_steps)
    tx = optax.adam(lr_scheduler)
    tx = optax.adam(1e-3)
    opt_state = tx.init(params)

    @jax.jit
    def loss_fn(params, key, xs, ys):
        logits, rollout = transformer.apply(params, key, xs)
        one_hot = jax.nn.one_hot(ys, num_classes=num_classes)
        loss = optax.softmax_cross_entropy(logits, one_hot).mean()
        return loss

    @jax.jit
    def calculate_metrics(params, key, xs, ys, k=5):
        logits, _ = transformer.apply(params, key, xs)
        classes = logits.argmax(axis=-1)
        accuracy = jnp.mean(classes == ys)

        top_k = jnp.argsort(logits, axis=-1)[:, -k:]
        hits = (ys == top_k.T).any(axis=0)
        top_k_accuracy = jnp.mean(hits)

        metrics = {
            "accuracy": accuracy,
            f"top_{k}_acc": top_k_accuracy,
        }
        return metrics

    @jax.jit
    def update(
        params: hk.Params,
        opt_state: optax.OptState,
        key: jax.random.PRNGKey,
        xs: tf.Tensor,
        ys: tf.Tensor
    ) -> t.Tuple[hk.Params, optax.OptState, jnp.ndarray]:
        loss, grads = jax.value_and_grad(loss_fn)(params, key, xs, ys)
        updates, opt_state = tx.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state, loss

    if not args.debug:
        mlflow.set_experiment("cifar_haiku")
        mlflow.start_run()
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("k", args.k)
        mlflow.log_param("heads", args.heads)
        mlflow.log_param("depth", args.depth)
        mlflow.log_param("patch_size", args.patch_size)
        mlflow.log_param("image_size", str(image_size))
        mlflow.log_param("dropout", args.dropout)
        mlflow.log_param("num_params", param_count)

    for e in range(args.epochs):
        step = 0
        metrics_dict = defaultdict(lambda: 0)
        desc = f"Train Epoch {e}"
        train_bar = tqdm(train_ds, total=len(train_ds), ncols=0, desc=desc)

        for xs, ys in train_bar:
            key = next(rng_seq)
            params, opt_state, loss = update(params, opt_state, key, xs, ys)
            metrics = calculate_metrics(params, key, xs, ys)
            metrics["loss"] = loss

            step += 1
            metrics_dict = update_metrics(step, metrics_dict, metrics)
            if step % show_every == 0:
                metrics_display = {k: str(v)[:4]
                                   for k, v in metrics_dict.items()}
                train_bar.set_postfix(**metrics_display)

        train_metrics = {f"train_{k}": float(v)
                         for k, v in metrics_dict.items()}
        if not args.debug:
            mlflow.log_metrics(train_metrics, step=e)

        step = 0
        metrics_dict = defaultdict(lambda: 0)
        desc = f"Valid Epoch {e}"
        val_bar = tqdm(val_ds, total=len(val_ds), ncols=0, desc=desc)

        for xs, ys in val_bar:
            key = next(rng_seq)
            loss = loss_fn(params, key, xs, ys)
            metrics = calculate_metrics(params, key, xs, ys)
            metrics["loss"] = loss

            step += 1
            metrics_dict = update_metrics(step, metrics_dict, metrics)
            if step % show_every == 0:
                metrics_display = {k: str(v)[:4]
                                   for k, v in metrics_dict.items()}
                val_bar.set_postfix(**metrics_display)

        val_metrics = {f"valid_{k}": float(v) for k, v in metrics_dict.items()}

        if not args.debug:
            mlflow.log_metrics(val_metrics, step=e)

        if e % save_every == 0 and not args.debug:
            pickle.dump(params, open("weights.pkl", "wb"))
            mlflow.log_artifact("weights.pkl", "weights")
            pickle.dump(opt_state, open("optimizer.pkl", "wb"))
            mlflow.log_artifact("optimizer.pkl", "optimizer")


if __name__ == "__main__":
    main()
