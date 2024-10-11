import base
import dataclasses
import optax
import enum

class Optimizer(enum.Enum):
    ADAM = "adam"
    SGD = "sgd"


@dataclasses.dataclass
class OptimizerConfig(base.MakeableConfig[optax.GradientTransformation]):
    optimizer: Optimizer
    learning_rate: float

    def make(self) -> optax.GradientTransformation:
        match self.optimizer:
            case Optimizer.ADAM:
                return optax.adam(self.learning_rate)
            case Optimizer.SGD:
                return optax.sgd(self.learning_rate)
