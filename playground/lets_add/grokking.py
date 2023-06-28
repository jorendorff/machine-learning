# Copied from https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/Non_Modular_Addition_Grokking_Tasks%20(1).ipynb
# and hacked into some semblance of order

import random
import time
import pickle
import os
from functools import *
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import einops
import tqdm
import matplotlib.pyplot as plt


# === Defining Transformer

# A helper class to get access to intermediate activations (inspired by Garcon)
# It's a dummy module that is the identity function by default
# I can wrap any intermediate activation in a HookPoint and get a convenient
# way to add PyTorch hooks
class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_hooks = []
        self.bwd_hooks = []

    def give_name(self, name):
        # Called by the model at initialisation
        self.name = name

    def add_hook(self, hook, dir='fwd'):
        # Hook format is fn(activation, hook_name)
        # Change it into PyTorch hook format (this includes input and output,
        # which are the same for a HookPoint)
        def full_hook(module, module_input, module_output):
            return hook(module_output, name=self.name)
        if dir=='fwd':
            handle = self.register_forward_hook(full_hook)
            self.fwd_hooks.append(handle)
        elif dir=='bwd':
            handle = self.register_backward_hook(full_hook)
            self.bwd_hooks.append(handle)
        else:
            raise ValueError(f"Invalid direction {dir}")

    def remove_hooks(self, dir='fwd'):
        if (dir=='fwd') or (dir=='both'):
            for hook in self.fwd_hooks:
                hook.remove()
            self.fwd_hooks = []
        if (dir=='bwd') or (dir=='both'):
            for hook in self.bwd_hooks:
                hook.remove()
            self.bwd_hooks = []
        if dir not in ['fwd', 'bwd', 'both']:
            raise ValueError(f"Invalid direction {dir}")

    def forward(self, x):
        return x


# --- Define network architecture
# I defined my own transformer from scratch so I'd fully understand each component
# - I expect this wasn't necessary or particularly important, and a bunch of this
# replicates existing PyTorch functionality

# Embed & Unembed
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        # W_E: [D, V] (D=model dimension, V=vocab size)
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        # x: [B, P] (B=batch size, P=example length)
        # W_E[:, x]: [D, B, P]
        # out: [B, P, D]
        return torch.einsum('dbp -> bpd', self.W_E[:, x])


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return (x @ self.W_U)


# Positional Embeddings
class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]


# LayerNorm
class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon = 1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x


# Attention
class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(num_heads, d_model, d_head) / np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()
        self.hook_result = HookPoint()

    def forward(self, x):
        # x: [B, P, D] (B=batch size, P=example length, D=dimension of model)
        # W_K, W_Q, W_V: [I, H, D] (I=number of attention heads, H=output dimension of each head)
        #
        # For each attention head, there's a single HxD linear combination that's applied to every token
        # of the input. Each token is size D. The output is size H. This is done for W_K, W_Q, and W_V.
        # The result is, for each example, for each attention head, for each token, a vector of size H.
        k = self.hook_k(torch.einsum('ihd,bpd->biph', self.W_K, x))
        q = self.hook_q(torch.einsum('ihd,bpd->biph', self.W_Q, x))
        v = self.hook_v(torch.einsum('ihd,bpd->biph', self.W_V, x))
        # Dot every key (B*I*P keys of size H) with the corresponding query (B*I*P queries of size H).
        # Result, for each example, for each attention head, is a PxP matrix of non-normalized cosine similarities.
        attn_scores_pre = torch.einsum('biph,biqh->biqp', k, q) # [B, I, P, P]
        # Replace everything above the q-p diagonal with -1e10. The triangle is visible when looking at
        # the last two dimensions; tril simply does the same to every such PxP square slice.
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])  # still [B, I, P, P]
        # Adjust for the contribution of H to variance, then apply softmax. Fine.
        # Now each row of each PxP square adds up to 1 and all elements are non-negative.
        attn_matrix = self.hook_attn(F.softmax(self.hook_attn_pre(attn_scores_masked / np.sqrt(self.d_head)), dim=-1))  # still [B, I, P, P]
        # For each batch, for each attention head, for each token, linearly combine the values
        # `v` using the weights in the `attn_matrix`.
        z = self.hook_z(torch.einsum('biph,biqp->biqh', v, attn_matrix))  # [B, I, P, H]
        # Lastly, each attention head has its own matrix to re-expand from H-vectors to D-vectors...
        result = self.hook_result(torch.einsum('idh,biqh->biqd', self.W_O, z)) # [B, I, P, H]
        # and then results from all attention heads are just added together to make the output.
        # It is wild that this mumbo jumbo produces anything useful at all.
        out = einops.reduce(result,
                             'batch index position model->batch position model',
                             'sum')  # [B, P, D]
        return out


# MLP Layers
class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        # self.ln = LayerNorm(d_mlp, model=self.model)
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()
        assert act_type in ['ReLU', 'GeLU']

    def forward(self, x):
        x = self.hook_pre(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        if self.act_type=='ReLU':
            x = F.relu(x)
        elif self.act_type=='GeLU':
            x = F.gelu(x)
        x = self.hook_post(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x


# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, attn_only, model):
        super().__init__()
        self.model = model
        # self.ln1 = LayerNorm(d_model, model=self.model)
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        # self.ln2 = LayerNorm(d_model, model=self.model)
        self.hook_attn_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_post = HookPoint()
        self.attn_only = attn_only
        if not self.attn_only:
            self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)
            self.hook_mlp_out = HookPoint()
            self.hook_resid_mid = HookPoint()

    def forward(self, x):
        x = (x + self.hook_attn_out(self.attn((self.hook_resid_pre(x)))))
        if not self.attn_only:
            x = (x + self.hook_mlp_out(self.mlp(self.hook_resid_mid(x))))
        return self.hook_resid_post(x)


# Full transformer
class Transformer(nn.Module):
    def __init__(self,
                 num_layers,
                 d_vocab,
                 d_model,
                 d_mlp,
                 d_head,
                 num_heads,
                 n_ctx,
                 act_type,
                 attn_only=False,
                 d_vocab_out=None,
                 use_pos=True):
        super().__init__()
        self.cache = {}
        self.attn_only = attn_only

        self.embed = Embed(d_vocab, d_model)
        self.use_pos = use_pos
        if self.use_pos:
            self.pos_embed = PosEmbed(n_ctx, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_mlp, d_head, num_heads, n_ctx, act_type, attn_only, model=[self]) for i in range(num_layers)])
        # self.ln = LayerNorm(d_model, model=[self])
        if d_vocab_out is None:
            d_vocab_out = d_vocab
        self.unembed = Unembed(d_vocab_out, d_model)

        for name, module in self.named_modules():
            if type(module)==HookPoint:
                module.give_name(name)

    def forward(self, x):
        x = self.embed(x)
        if self.use_pos:
            x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x

    def hook_points(self):
        return [module for name, module in self.named_modules() if 'hook' in name]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks('fwd')
            hp.remove_hooks('bwd')

    def cache_all(self, cache, incl_bwd=False):
        # Caches all activations wrapped in a HookPoint
        def save_hook(tensor, name):
            cache[name] = tensor.detach()
        def save_hook_back(tensor, name):
            cache[name+'_grad'] = tensor[0].detach()
        for hp in self.hook_points():
            hp.add_hook(save_hook, 'fwd')
            if incl_bwd:
                hp.add_hook(save_hook_back, 'bwd')


# === 5 Digit Addition

# --- Setup

# Model
num_layers = 1
d_vocab = 12
d_vocab_out = 10
d_model = 512 #@param
num_heads = 4
d_head = d_model // num_heads
d_mlp = 4 * d_model
seed = 129000 #@param

# Data
num_digits =  5#@param
n_ctx = 3*num_digits + 3
act_type = 'ReLU'
batch_size = 64 #@param
is_finite = False #@param
num_data = 750 #@param

# Optimizer
lr = 1e-4 #@param
weight_decay = 0.1 #@param
num_epochs = 3000 #@param

# Misc
checkpoint_models = False #@param
checkpoint_every = 50 #@param


PLUS_INDEX = 10
EQUALS_INDEX = 11


def to_numpy(tensor, flat=False):
    if type(tensor)!=torch.Tensor:
        return tensor
    if flat:
        return tensor.flatten().detach().cpu().numpy()
    else:
        return tensor.detach().cpu().numpy()


def tokens_to_string(tokens):
    tokens = to_numpy(tokens)
    x = "".join([str(i) for i in tokens[:5]])
    y = "".join([str(i) for i in tokens[6:11]])
    z = "".join([str(i) for i in tokens[12:]])
    return f"  {x}\n +{y}\n={z}"


def string_to_tokens(string, batch=False):
    lookup = {str(i):i for i in range(10)}
    lookup['+']=10
    lookup['=']=11
    tokens = [lookup[i] for i in string if i not in '\n ']
    if batch:
        return torch.tensor(tokens)[None, :]
    else:
        return torch.tensor(tokens)


def data_generator(batch_size, num_digits, seed):
    torch.manual_seed(seed)
    while True:
        batch = torch.zeros((batch_size, 3*num_digits+3)).to(torch.int64)
        x = torch.randint(0, 10, (batch_size, num_digits))
        y = torch.randint(0, 10, (batch_size, num_digits))
        batch[:, :num_digits] = x
        batch[:, num_digits] = PLUS_INDEX
        batch[:, 1+num_digits:1+num_digits*2] = y
        batch[:, 1+num_digits*2] = EQUALS_INDEX
        carries = [torch.zeros((batch_size,)).to(torch.int64)]
        for i in range(num_digits):
            carry = carries[-1]
            digit_sum = (batch[:, num_digits-1-i]+batch[:, 2*num_digits-i]+carry)
            batch[:, -1-i] = (digit_sum % 10)
            carry = (digit_sum >= 10).to(torch.int64)
            carries.append(carry)
        batch[:, -1-num_digits] = carries[-1]
        carries = torch.stack(carries, axis=1)
        yield batch.cuda(), carries.cuda()


if is_finite:
    test_ds = data_generator(batch_size, num_digits, seed)
    train_ds = data_generator(num_data, num_digits, seed)
    train_tokens, train_carries = next(train_ds)
else:
    ds = data_generator(batch_size, num_digits, seed)


torch.manual_seed(seed)
model = Transformer(num_layers=num_layers,
                    d_vocab=d_vocab,
                    d_model=d_model,
                    d_mlp=d_mlp,
                    d_head=d_head,
                    num_heads=num_heads,
                    n_ctx=n_ctx,
                    act_type=act_type,
                    d_vocab_out=d_vocab_out)
model.to('cuda')
optimizer = optim.AdamW(model.parameters(),
                        lr=lr,
                        weight_decay=weight_decay,
                        betas=(0.9, 0.98))
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))


def get_pred_log_probs(logits, tokens):
    trunc_logits = logits[:, -(num_digits+2):-1]
    ans_tokens = tokens[:, -(num_digits+1):]
    log_probs = F.log_softmax(trunc_logits.to(torch.float64), dim=-1)
    pred_log_probs = torch.gather(log_probs, -1, ans_tokens[:, :, None])[..., 0]
    return pred_log_probs


def loss_fn(logits, tokens):
    return -get_pred_log_probs(logits, tokens).mean()


import torchinfo
torchinfo.summary(model, input_data = train_tokens)


# --- Training

if is_finite:
    train_losses = []
    ptl_train_list = []
    test_losses = []
    ptl_test_list = []
    # per_token_losses_list = []
    # sds=[]
    # epochs = [0]
    # sds.append(model.state_dict())
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_logits = model(train_tokens)
        per_token_losses_train = -get_pred_log_probs(train_logits, train_tokens).mean(0)
        ptl_train_list.append(to_numpy(per_token_losses_train))
        train_loss = per_token_losses_train.mean()
        train_loss.backward()
        train_losses.append(train_loss.item())

        test_tokens, _ = next(test_ds)
        test_logits = model(test_tokens)
        per_token_losses_test = -get_pred_log_probs(test_logits, test_tokens).mean(0)
        ptl_test_list.append(to_numpy(per_token_losses_test))
        test_loss = per_token_losses_test.mean()
        test_losses.append(test_loss.item())

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print(epoch, train_loss.item(), test_loss.item())
else:
    train_losses = []
    per_token_losses_list = []
    sds=[]
    epochs = [0]
    sds.append(model.state_dict())
    for epoch in tqdm.tqdm(range(num_epochs)):
        tokens, carry = next(ds)
        logits = model(tokens)
        per_token_losses = -get_pred_log_probs(logits, tokens).mean(0)
        per_token_losses_list.append(to_numpy(per_token_losses))
        loss = per_token_losses.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())
        if epoch % 100 == 0:
            print(epoch, loss.item())
        if checkpoint_models:
            if (epoch+1) % (checkpoint_every) == 0:
                sds.append(model.state_dict())
                epochs.append(epoch+1)


# here, try some examples to see it working
TOKENS='0123456789+='
assert TOKENS[PLUS_INDEX] == '+'
assert TOKENS[EQUALS_INDEX] == '='
while True:
    line = input("> ")
    line = line.replace(' ', '')
    if len(line) != 12 or not all(c in TOKENS for c in line):
        print("couldn't parse -- type something like:   12345 + 54459 =")
        continue
    tokens = [TOKENS.index(c) for c in line]
    while len(tokens) < 18:
        logits = model(torch.tensor([tokens]).to('cuda'))[0]
        prediction = logits.argmax(dim=-1)
        tokens.append(prediction[len(tokens) - 1])
    print(''.join(TOKENS[i] for i in tokens))
