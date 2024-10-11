import dataclasses
import optimizers
import base
import models


@dataclasses.dataclass
class TrainingConfig:
    optimizer_config: optimizers.OptimizerConfig
    num_train_steps: int


@dataclasses.dataclass
class TransformerConfig(base.MakeableConfig[models.Transformer]):
    vocab_size: int
    d_model: int
    sequence_length: int
    num_blocks: int
    num_heads: int
    ffw_multiplier: int

    def make(self) -> models.Transformer:
        return models.Transformer(
            vocab_size=self.vocab_size,
            sequence_length=self.sequence_length,
            d_model=self.d_model,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            ffw_multiplier=self.ffw_multiplier,
        )
