import jax
import haiku as hk
import jax.numpy as jnp

from einops import rearrange, repeat


class SelfAttention(hk.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.k = k
        self.heads = heads

        self.to_queries = hk.Linear(k*heads, with_bias=False)
        self.to_keys = hk.Linear(k*heads, with_bias=False)
        self.to_values = hk.Linear(k*heads, with_bias=False)
        self.unify_heads = hk.Linear(k)

    def __call__(self, x):
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
        dot = jax.nn.softmax(dot, axis=2)

        out = rearrange(jax.lax.batch_matmul(dot, values),
                        "(b h) t k -> b t (h k)", h=h)
        attention = self.unify_heads(out)

        return attention


class TransformerBlock(hk.Module):
    def __init__(self, k, heads, dropout):
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

    def __call__(self, x, inference=False):
        dropout = 0. if inference else self.dropout

        x = self.layer_norm_1(self.attention(x)) + x

        key1 = hk.next_rng_key()
        key2 = hk.next_rng_key()

        forward = self.linear_1(x)
        forward = jax.nn.gelu(forward)
        forward = hk.dropout(key1, dropout, forward)
        forward = self.linear_2(forward)
        forward = self.layer_norm_2(forward + x)
        out = hk.dropout(key2, dropout, forward)

        return out


class VisionTransformer(hk.Module):
    def __init__(self, k, heads, depth, num_classes, patch_size, image_size, dropout):
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
        self.blocks = hk.Sequential([
            TransformerBlock(self.k, self.heads, dropout) for _ in range(self.depth)
        ])
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
        x = self.blocks(x)
        x = x[:, 0]
        x = self.classification(x)

        return x
