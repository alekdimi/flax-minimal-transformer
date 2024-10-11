from absl.testing import absltest

import jax
import jax.numpy as jnp
import models


class ModelsTest(absltest.TestCase):

    def test_mlp(self):
        mlp = models.MLP(num_hidden=5, num_outputs=3, use_bias=False)
        key = jax.random.PRNGKey(0)
        input = jnp.ones((16, 3))
        params = mlp.init(key, input)
        output = mlp.apply(params, input)
        self.assertEqual(output.shape, input.shape)

    def test_block(self):
        B, T, D = 2, 4, 8
        block = models.Block(d_model=D, ffw_multiplier=4, num_heads=2)
        key = jax.random.PRNGKey(0)
        params = block.init(key, jnp.ones((B, T, D)))
        x = jnp.ones((B, T, D))
        y = block.apply(params, x)
        self.assertEqual(y.shape, (B, T, D))

    def test_transformer(self):
        batch_size, sequence_length, vocab_size = 2, 4, 6
        transformer = models.Transformer(
            vocab_size=vocab_size,
            sequence_length=sequence_length,
            d_model=8,
            num_blocks=2,
            num_heads=2,
            ffw_multiplier=4,
        )
        tokens = jax.random.randint(
            jax.random.PRNGKey(0), (batch_size, sequence_length), 0, vocab_size
        )
        params = transformer.init(jax.random.PRNGKey(0), tokens)
        output = transformer.apply(params, tokens)
        self.assertEqual(output.shape, (batch_size, sequence_length, vocab_size))


if __name__ == "__main__":
    absltest.main()
