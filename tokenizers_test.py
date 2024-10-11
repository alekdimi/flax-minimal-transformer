from absl.testing import absltest
from absl.testing import parameterized

import tiktoken
import tokenizers


class TokenizerTest(parameterized.TestCase):

    @parameterized.parameters(("gpt2",))
    def test_tokenizer(self, model: str):
        tokenizer = tokenizers.TokenizerConfig(model=model).make()
        self.assertIsInstance(tokenizer, tiktoken.Encoding)
        tokens = tokenizer.encode("Hello, world!")
        self.assertEqual(tokens, [15496, 11, 995, 0])


if __name__ == "__main__":
    absltest.main()
