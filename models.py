import base
import dataclasses
import flax.linen as nn
import jax
import jax.numpy as jnp


class Linear(nn.Module):
    """Linear layer."""

    features: int
    use_bias: bool

    def setup(self):
        self.linear = nn.Dense(features=self.features, use_bias=self.use_bias)

    def __call__(self, x):
        return self.linear(x)


class MLP(nn.Module):
    """Multi-layer perceptron with one hidden layer and GeLU activation."""

    num_hidden: int
    num_outputs: int
    use_bias: bool

    def setup(self):
        self.layer1 = nn.Dense(features=self.num_hidden, use_bias=self.use_bias)
        self.layer2 = nn.Dense(features=self.num_outputs, use_bias=self.use_bias)

    def __call__(self, x):
        x = self.layer1(x)
        x = jax.nn.gelu(x)
        return self.layer2(x)


class Block(nn.Module):
    """Transformer block."""

    d_model: int
    num_heads: int
    ffw_multiplier: int

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
        )
        self.norm2 = nn.LayerNorm()
        self.ffw = MLP(
            num_hidden=self.d_model * self.ffw_multiplier,
            num_outputs=self.d_model,
            use_bias=False,
        )

    def __call__(self, x):
        x = self.norm1(x)
        x = self.attn(x, x, x)
        x = self.norm2(x)
        x = self.ffw(x)
        return x


class Transformer(nn.Module):
    """Decoder only transformer."""

    vocab_size: int
    sequence_length: int
    d_model: int
    num_blocks: int
    num_heads: int
    ffw_multiplier: int

    def setup(self):
        self.token_embedder = nn.Embed(
            num_embeddings=self.vocab_size, features=self.d_model
        )
        self.position_embedder = nn.Embed(
            num_embeddings=self.sequence_length, features=self.d_model
        )
        self.blocks = [
            Block(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ffw_multiplier=self.ffw_multiplier,
            )
            for _ in range(self.num_blocks)
        ]

    def __call__(self, x):
        x = self.token_embedder(x)
        positions = self.position_embedder(jnp.arange(x.shape[1]))
        x += positions
        for block in self.blocks:
            x = block(x)
        x = self.token_embedder.attend(x)
        return x
