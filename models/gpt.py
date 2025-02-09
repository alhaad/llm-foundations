from flax import nnx
import jax
import jax.numpy as jnp
import functools

class SelfAttentionHead(nnx.Module):
    def __init__(self, n_embed, head_dim, rngs: nnx.Rngs):
        self.query = nnx.Linear(n_embed, head_dim, rngs=rngs, use_bias=False)
        self.key = nnx.Linear(n_embed, head_dim, rngs=rngs, use_bias=False)
        self.value = nnx.Linear(n_embed, head_dim, rngs=rngs, use_bias=False)
    
    def __call__(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)

        attn = jnp.einsum('btd,bTd->btT', q, k) / jnp.sqrt(16)

        tril = jnp.tril(jnp.ones((T, T)))
        attn = jnp.where(tril[:T, :T] == 0, float('-inf'), attn)
        attn = jax.nn.softmax(attn)

        v = self.value(x)
        return attn @ v
    
class MultiHeadAttention(nnx.Module):
    def __init__(self, n_embed, n_head, head_dim, rngs: nnx.Rngs):
        self.heads = [SelfAttentionHead(n_embed, head_dim, rngs) for _ in range(n_head)]
        self.proj = nnx.Linear(n_embed, n_embed, rngs=rngs)

    def __call__(self, x):
        x = jnp.concatenate([head(x) for head in self.heads], axis=-1)
        return self.proj(x)
    
class FeedForward(nnx.Module):
    def __init__(self, n_embed, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(n_embed, 4 * n_embed, rngs=rngs)
        self.fc2 = nnx.Linear(4 * n_embed, n_embed, rngs=rngs)
    
    def __call__(self, x):
        return self.fc2(jax.nn.relu(self.fc1(x)))

class Block(nnx.Module):
    def __init__(self, n_embed, n_head, rngs: nnx.Rngs):
        self.sa_heads = MultiHeadAttention(n_embed, n_head, n_embed // n_head, rngs=rngs)
        self.ffwd = FeedForward(n_embed, rngs=rngs)
        self.ln1 = nnx.LayerNorm(n_embed, rngs=rngs)
        self.ln2 = nnx.LayerNorm(n_embed, rngs=rngs)

    def __call__(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nnx.Module):
    def __init__(self, block_size, vocab_size, n_embed, n_head, n_blocks, rngs: nnx.Rngs):
        self.block_size = block_size
        self.token_embedding_table = nnx.Embed(num_embeddings=vocab_size, features=n_embed, rngs=rngs)
        self.position_embedding_table = nnx.Embed(num_embeddings=self.block_size, features=n_embed, rngs=rngs)
        self.blocks = nnx.Sequential(*[Block(n_embed, n_head, rngs) for _ in range(n_blocks)])
        # lm_head is tied to token_embedding_table.


    def __call__(self, x):
        B, T = x.shape
        x = self.token_embedding_table(x) + self.position_embedding_table(jnp.arange(T))
        x = self.blocks(x)
        logits = self.token_embedding_table.attend(x)
        return logits