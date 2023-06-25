import math, os
from tempfile import TemporaryDirectory
from typing import Iterable, Tuple
import time

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [n, batch_size, embedding_dim]
        # Note: Throughout, `n` means the sequence length for a single input to the model, in tokens.
        return x + self.pe[:x.size(0)]


class TransformerModel(nn.Module):
    def __init__(self, ntokens: int):
        dim_model = 200  # dimension of embeddings (and residual links, i think)
        dim_feedforward = 200 # dimensions of the feedforward network model in the TransformerEncoder

        super().__init__()
        self.ntokens = ntokens
        self.dim_model = dim_model

        self.embedding = nn.Embedding(ntokens, dim_model)
        self.pos_encoder = PositionalEncoding(dim_model, dropout=0.2)
        encoder_layer = TransformerEncoderLayer(
            d_model=dim_model, nhead=2, dim_feedforward=dim_feedforward, dropout=0.2)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
        self.linear = nn.Linear(dim_model, ntokens)

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        # src shape: [n, k], src_mask shape: [n, n], output shape: [n, k, ntokens]
        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = self.pos_encoder(src)
        return self.linear(self.transformer(src, src_mask))


assert torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


def batchify(lines: Iterable[str], k: int) -> Tensor:
    # tokenize `lines` and concat into one huge flat Tensor
    data = [torch.tensor(vocab(tokenizer(line)), dtype=torch.long) for line in lines]
    data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    # split the data into `k` columns, for parallelism
    seq_len = data.size(0) // k
    data = data[:seq_len * k]
    data = data.view(k, seq_len).t().contiguous()
    return data.to(DEVICE) # shape: [len(data)//k, k]


train_iter, val_iter, test_iter = WikiText2()
train_data = batchify(train_iter, 20)  # shape `[seq_len//20, 20]`
val_data = batchify(val_iter, 10)
test_data = batchify(test_iter, 10)


bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    # source shape: [full_seq_size, k]; return shapes: [n, k] and [n * k]

    # why are data and target different shapes? it's inessential - CrossEntropyLoss wants to see a
    # 1-dimensional array of answers per batch, i guess. so we must flatten both the model output
    # (`output`, below) and the expected output (`target`, here).
    n = min(bptt, len(source) - 1 - i)
    data = source[i:i+n]
    target = source[i+1:i+1+n].reshape(-1)
    return data, target


def train_one_epoch(epoch: int, model: nn.Module, criterion: nn.Module, train_data: Tensor, scheduler, optimizer: torch.optim.SGD) -> None:
    model.train()  # turn on train mode
    total_loss = 0.0
    log_interval = 200
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)): # why `- 1` here?
        data, targets = get_batch(train_data, i)
        output = model(data)
        output_flat = output.view(-1, model.ntokens)
        loss = criterion(output_flat, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, criterion: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, model.ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)


model = TransformerModel(len(vocab)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
best_val_loss = float('inf')

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()
    train_one_epoch(epoch, model, criterion, train_data, scheduler, optimizer)
    val_loss = evaluate(model, criterion, val_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
        f'eval loss {val_loss:5.2f} | eval perplexity {val_ppl:8.2f}')
    print('-' * 89)
    # torch.save(model.state_dict(), filepath)
    scheduler.step()

test_loss = evaluate(model, criterion, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test perplexity {test_ppl:8.2f}')
print('=' * 89)
