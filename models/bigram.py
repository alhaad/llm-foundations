from flax import nnx
import jax
import jax.numpy as jnp

class BigramLanguageModel(nnx.Module):
    def __init__(self, vocab_size, rngs: nnx.Rngs):
        self.rngs = rngs
        self.token_embedding_table = nnx.Embed(num_embeddings=vocab_size, features=vocab_size, rngs=rngs)

    def __call__(self, x):
        logits = self.token_embedding_table(x)
        return logits
    
    def generate(self, x, length): # x has the shape (batch_size, block_size)
        for i in range(length):
            logits = self(x)
            next_token = jax.random.categorical(self.rngs.next(), logits[:, -1])
            x = jnp.concatenate([x, next_token[:, None]], axis=1)
        return x