import flax.linen as nn
import jax


class Linear(nn.Module):
    """Linear layer."""
    features: int
    use_bias: bool

    def setup(self):
        self.linear = nn.Dense(features=self.features, use_bias=self.use_bias)

    def __call__(self, x):
        return self.linear(x)


class MLP(nn.Module):
    """Multi-layer perceptron."""
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
