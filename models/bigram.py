from flax import nnx
import jax
import jax.numpy as jnp

class BigramLanguageModel(nnx.Module):
    def __init__(self, vocab_size, rngs: nnx.Rngs):
        self.token_embedding_table = nnx.Embed(num_embeddings=vocab_size, features=vocab_size, rngs=rngs)

    def __call__(self, x):
        logits = self.token_embedding_table(x)
        return logits