import dataclasses
import optimizers

@dataclasses.dataclass
class TrainingConfig:
    optimizer: optimizers.OptimizerConfig
    num_train_steps: int
