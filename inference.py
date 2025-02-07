import jax
import jax.numpy as jnp
from flax import nnx

# Load from checkpoint

# Hyperparameters
seed = 1337
batch_size = 32
block_size = 8
n_embed = 32
n_head = 4
n_blocks = 4

# Prepare data
with open('notebooks/data/tinyshakespeare') as f:
    text = f.read()

# Tokenzier
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('out/shakespeare_tokenizer_model.model')
vocab_size = sp.get_piece_size()
encode, decode = sp.encode, sp.decode

from models.gpt import GPT
abstract_model = nnx.eval_shape(lambda: GPT(block_size, vocab_size, n_embed, n_head, n_blocks, rngs=nnx.Rngs(seed)))
graphdef, abstract_state = nnx.split(abstract_model)

import orbax.checkpoint as ocp
import os
checkpointer = ocp.StandardCheckpointer()
state_restored = checkpointer.restore(os.getcwd() + '/out/tinyshakespeare-gpt/state', abstract_state)
model = nnx.merge(graphdef, state_restored)

# Generation
rngs = nnx.Rngs(0)
def generate(x, length):
    for i in range(length):
        logits = model(x[:, -block_size:])
        next_token = jax.random.categorical(rngs.next(), logits[:, -1])
        x = jnp.concatenate([x, next_token[:, None]], axis=1)
    return x
print([decode(row.tolist()) for row in generate(jnp.zeros((1, 1), dtype=jnp.int32), 500)][0])