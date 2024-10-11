import dataclasses
import tiktoken

@dataclasses.dataclass
class TokenizerConfig:
    model: str

    def make(self) -> tiktoken.Encoding:
        return tiktoken.encoding_for_model(self.model)
