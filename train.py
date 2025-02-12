
from flax import nnx
import jax
import jax.numpy as jnp
import optax
import tqdm

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
spm.SentencePieceTrainer.train(input='notebooks/data/tinyshakespeare',
                               model_prefix='out/shakespeare_tokenizer_model',
                               vocab_size=1000,
                               character_coverage=1.0,
                               model_type='unigram',
                               remove_extra_whitespaces=False,
                               user_defined_symbols=["\n", "\r"])

sp = spm.SentencePieceProcessor()
sp.load('out/shakespeare_tokenizer_model.model')
vocab_size = sp.get_piece_size()
encode, decode = sp.encode, sp.decode

data = jnp.array(encode(text), dtype=jnp.int32)
train_data = data[: int(.9 * len(data))]
val_data = data[int(.9 * len(data)):]

# Data loader
dynamic_slice_vmap = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))
@jax.jit
def get_batch(data, key):
    ix = jax.random.randint(key, shape=(batch_size, 1), minval=0, maxval=len(data) - block_size)
    x = dynamic_slice_vmap(data, ix, (block_size,))
    y = dynamic_slice_vmap(data, ix + 1, (block_size,))
    return x, y

# Random number genration
key = jax.random.key(seed)
rngs = nnx.Rngs(key)

# Model
from models.gpt import GPT
model = GPT(block_size, vocab_size, n_embed, n_head, n_blocks, rngs)

# Loss function
def loss_fn(model, x, targets):
        logits = model(x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

# Training
# The following code is slow. See https://flax.readthedocs.io/en/latest/guides/performance.html#performance-considerations
# It is supposed to be fixed.
# @nnx.jit
# def train_step(model, optimizer, xb, yb):
#     grads = (nnx.grad(loss))(model, xb, yb)
#     optimizer.update(grads)

# def train(key, model):
#     optimizer = nnx.Optimizer(model, optax.adamw(1e-3))
#     for i in tqdm.trange(10000):
#         key, subkey = jax.random.split(key)
#         xb, yb = get_batch(train_data, subkey)
#         train_step(model, optimizer, xb, yb)
# train(key, model)

# The code below is a workaround for the slow training code above.
optimizer = nnx.Optimizer(model, optax.adamw(1e-3))
metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
graphdef, state = nnx.split((model, optimizer, metrics))
@jax.jit
def train_step(graphdef, state, xb, yb):
    model, optimizer, metrics = nnx.merge(graphdef, state)
    grads = (nnx.grad(loss_fn))(model, xb, yb)
    optimizer.update(grads)
    _, state = nnx.split((model, optimizer, metrics))
    return state

for i in tqdm.trange(10000):
    key, subkey = jax.random.split(key)
    xb, yb = get_batch(train_data, subkey)
    state = train_step(graphdef, state, xb, yb)
nnx.update((model, optimizer, metrics), state)

train_xb, train_yb = get_batch(train_data, key)
print(loss_fn(model, train_xb, train_yb))

val_xb, val_yb = get_batch(val_data, key)
print(loss_fn(model, val_xb, val_yb))

# Save model
import orbax.checkpoint as ocp
import os
ckpt_dir = ocp.test_utils.erase_and_create_empty(os.getcwd() + '/out/tinyshakespeare-gpt/')
_, state = nnx.split(model)
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(ckpt_dir / 'state', state)
checkpointer.wait_until_finished()