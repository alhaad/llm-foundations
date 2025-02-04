from flax import nnx
import jax
import jax.numpy as jnp

class Lookback(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.rngs = rngs
    
    def __call__(self, x):
        B, T, C = x.shape
        tril = jnp.tril(jnp.ones((T, T)))
        attn = jnp.zeros((T, T))
        attn = jnp.where(tril == 0, float('-inf'), attn)
        attn = jax.nn.softmax(attn)
        return attn @ x

class NgramLanguageModel(nnx.Module):
    def __init__(self, block_size, vocab_size, n_embed, rngs: nnx.Rngs):
        self.block_size = block_size
        self.rngs = rngs
        self.token_embedding_table = nnx.Embed(num_embeddings=vocab_size, features=n_embed, rngs=rngs)
        self.position_embedding_table = nnx.Embed(num_embeddings=block_size, features=n_embed, rngs=rngs)
        self.lookback = Lookback(rngs)
        self.lm_head = nnx.Linear(n_embed, vocab_size, rngs=rngs)

    def __call__(self, x):
        B, T = x.shape
        x = self.token_embedding_table(x) + self.position_embedding_table(jnp.arange(T))
        x = self.lookback(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, x, length):
        for i in range(length):
            logits = self(x[:, -self.block_size:])
            next_token = jax.random.categorical(self.rngs.next(), logits[:, -1])
            x = jnp.concatenate([x, next_token[:, None]], axis=1)
        return x