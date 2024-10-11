import base
import dataclasses
import tiktoken


@dataclasses.dataclass
class TokenizerConfig(base.MakeableConfig[tiktoken.Encoding]):
    model: str

    def make(self) -> tiktoken.Encoding:
        return tiktoken.encoding_for_model(self.model)
