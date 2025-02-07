def get_encode_decode(text):
    vocab = sorted(list(set(text)))
    itos = {i:s for i,s in enumerate(vocab)}
    stoi = {s:i for i,s in enumerate(vocab)}
    encode = lambda x: [stoi[s] for s in x]
    decode = lambda x: ''.join([itos[i] for i in x])
    return encode, decode