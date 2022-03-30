import jax
import typing as t
import haiku as hk
import numpy as np
import jax.numpy as jnp

from einops import rearrange, repeat, reduce


class SelfAttention(hk.Module):
    def __init__(self, k: int, heads: int):
        super().__init__()
        self.k = k
        self.heads = heads

        self.to_queries = hk.Linear(k*heads, with_bias=False)
        self.to_keys = hk.Linear(k*heads, with_bias=False)
        self.to_values = hk.Linear(k*heads, with_bias=False)
        self.unify_heads = hk.Linear(k)

    def __call__(self, x: jnp.ndarray):
        h = self.heads
        k = self.k

        queries = self.to_queries(x)
        keys = self.to_keys(x)
        values = self.to_values(x)

        queries = rearrange(queries, "b t (k h)  -> (b h) t k", h=h)
        keys = rearrange(keys, "b t (k h) -> (b h) t k", h=h)
        values = rearrange(values, "b t (k h) -> (b h) t k", h=h)

        queries = queries / (k ** (1/4))
        keys = keys / (k ** (1/4))

        dot = jax.lax.batch_matmul(queries, rearrange(keys, "b t k -> b k t"))

        # send attention heads as additional output
        heads = rearrange(dot, "(b h) t k -> b h t k", h=h)
        dot = jax.nn.softmax(dot, axis=2)

        out = rearrange(jax.lax.batch_matmul(dot, values),
                        "(b h) t k -> b t (h k)", h=h)
        attention = self.unify_heads(out)

        return attention, heads


class TransformerBlock(hk.Module):
    def __init__(self, k: int, heads: int, dropout: float):
        super().__init__()
        self.k = k
        self.heads = heads
        self.dropout = dropout

        self.attention = SelfAttention(self.k, self.heads)
        self.layer_norm_1 = hk.LayerNorm(
            axis=[-2, -1], create_scale=True, create_offset=True)
        self.linear_1 = hk.Linear(4*self.k)
        self.linear_2 = hk.Linear(self.k)

        self.layer_norm_2 = hk.LayerNorm(
            axis=[-2, -1], create_scale=True, create_offset=True)

    def __call__(self, x: jnp.ndarray, inference=False):
        dropout = 0. if inference else self.dropout
        x, heads = self.attention(x)
        x = self.layer_norm_1(x) + x

        key1 = hk.next_rng_key()
        key2 = hk.next_rng_key()

        forward = self.linear_1(x)
        forward = jax.nn.gelu(forward)
        forward = hk.dropout(key1, dropout, forward)
        forward = self.linear_2(forward)
        forward = self.layer_norm_2(forward + x)
        out = hk.dropout(key2, dropout, forward)

        return out, heads


class VisionTransformer(hk.Module):
    def __init__(
        self,
        k,
        heads: int,
        depth: int,
        num_classes: int,
        patch_size: int,
        image_size: t.Tuple[int, int],
        dropout: float
    ):
        super().__init__()
        self.k = k
        self.heads = heads
        self.depth = depth
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.image_size = image_size
        self.dropout = dropout

        # Patch embedding is just a dense layer mapping a flattened patch to another array
        self.token_emb = hk.Linear(self.k)
        self.blocks = [
            TransformerBlock(self.k, self.heads, dropout) for _ in range(self.depth)
        ]
        self.classification = hk.Linear(self.num_classes)
        height, width = image_size
        num_patches = (height // patch_size) * (width // patch_size) + 1

        self.pos_emb = hk.Embed(vocab_size=num_patches, embed_dim=self.k)
        self.cls_token = hk.get_parameter(
            "cls", shape=[k], init=hk.initializers.RandomNormal())

        self.classification = hk.Sequential([
            hk.LayerNorm(axis=[-2, -1], create_scale=True, create_offset=True),
            hk.Linear(self.num_classes),
        ])

    def __call__(self, x, inference=False):
        dropout = 0. if inference else self.dropout

        batch_size = x.shape[0]
        x = rearrange(x, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
                      p1=self.patch_size, p2=self.patch_size)
        tokens = self.token_emb(x)

        cls_token = repeat(self.cls_token, "k -> b 1 k", b=batch_size)
        combined_tokens = jnp.concatenate([cls_token, tokens], axis=1)
        positions = jnp.arange(combined_tokens.shape[1])
        pos_emb = self.pos_emb(positions)
        x = pos_emb + combined_tokens
        x = hk.dropout(hk.next_rng_key(), dropout, x)

        attention_heads = []
        for block in self.blocks:
            x, heads = block(x)
            attention_heads.append(heads)

        rollout = attention_rollout(attention_heads, head_fusion="max", discard_ratio=0.5)

        x = x[:, 0]
        x = self.classification(x)

        return x, rollout


def attention_rollout(
    attention_heads: t.List[jnp.ndarray],
    head_fusion: str,
    discard_ratio: float = 0,
) -> jnp.ndarray:
    batch, _, tokens, _ = attention_heads[0].shape
    rollout = repeat(jnp.eye(tokens), "h1 h2 -> b h1 h2", b=batch)

    # Multiply attention in each block together
    for attention in attention_heads:
        if head_fusion == "mean":
            attention_heads_fused = attention.mean(axis=1)
        elif head_fusion == "max":
            attention_heads_fused = attention.max(axis=1)
        elif head_fusion == "min":
            attention_heads_fused = attention.min(axis=1)
        else:
            raise ValueError("Attention head fusion type Not supported")

        if discard_ratio != 0:
            flat_attn = rearrange(attention_heads_fused, "b h w -> b (h w)")
            # Take the top percentile across the last axis
            threshold = jnp.percentile(flat_attn, (1 - discard_ratio) * 100, axis=-1, keepdims=True)

            # Mask to keep the class token
            cls_indices = np.zeros(flat_attn.shape)
            cls_indices[:, 0] = 1
            cls_indices = jnp.array(cls_indices)

            # Keep values that are in the top percentile or are the cls indices
            keep_mask = jnp.logical_or(flat_attn > threshold, cls_indices)
            flat_attn = jnp.where(keep_mask, flat_attn, 0)

            filtered_attn = rearrange(flat_attn, "b (h w) -> b h w", h=tokens, w=tokens)
        else:
            filtered_attn = attention_heads_fused

        # Compute attention rollout
        identity = repeat(jnp.eye(tokens), "x y -> b x y", b=batch)
        a = (filtered_attn + 1.0 * identity) / 2
        # Normalize values over embedding axis
        a = a / reduce(a, "b h1 h2 -> b h1 1", "sum")
        rollout = jax.lax.batch_matmul(a, rollout)

    masks = rollout[:, 0, 1:]
    width = int((tokens - 1) ** 0.5)
    masks = rearrange(masks, "b (w1 w2) -> b w1 w2", w1=width, w2=width)
    masks = masks / reduce(masks, "b w1 w2 -> b 1 1", "max")

    return rollout
