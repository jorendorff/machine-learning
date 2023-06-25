import math, os
from tempfile import TemporaryDirectory
from typing import Tuple
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
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [seq_len, batch_size, embedding_dim]
        return self.dropout(x + self.pe[:x.size(0)])


class TransformerModel(nn.Module):
    def __init__(self, ntokens: int):
        dim_embedding = 200  # embedding dimension
        dim_feedforward = 200 # dimensions of the feedforward network model in the TransformerEncoder

        super().__init__()
        self.ntokens = ntokens
        self.dim_embedding = dim_embedding

        self.embedding = nn.Embedding(ntokens, dim_embedding)
        self.pos_encoder = PositionalEncoding(dim_embedding, dropout=0.2)
        encoder_layer = TransformerEncoderLayer(
            d_model=dim_embedding, nhead=2, dim_feedforward=dim_feedforward, dropout=0.2)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)
        self.linear = nn.Linear(dim_feedforward, ntokens)

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        # src shape: [n, k], src_mask shape: [n, n], output shape: [n, k, ntokens]
        src = self.embedding(src) * math.sqrt(self.dim_embedding)
        src = self.pos_encoder(src)
        return self.linear(self.transformer(src, src_mask))


assert torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_data():
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Convert list of strings (lines of text) into one huge flat Tensor."""
        data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def batchify(data: Tensor, k: int) -> Tensor:
        # data shape: [N], return shape: [N//k, k]
        seq_len = data.size(0) // k
        data = data[:seq_len * k]
        data = data.view(k, seq_len).t().contiguous()
        return data.to(DEVICE)


    # ``train_iter`` was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = batchify(data_process(train_iter), 20)  # shape `[seq_len//2, 20]`
    val_data = batchify(data_process(val_iter), 10)
    test_data = batchify(data_process(test_iter), 10)

    return vocab, train_data, val_data, test_data


bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def make_model(vocab):
    return 


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


def train(model: nn.Module, criterion: nn.Module, train_data: Tensor, val_data: Tensor, scheduler, optimizer: torch.optim.SGD):
    best_val_loss = float('inf')
    epochs = 3

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train_one_epoch(epoch, model, criterion, train_data, scheduler, optimizer)
            val_loss = evaluate(model, criterion, val_data)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                f'eval loss {val_loss:5.2f} | eval perplexity {val_ppl:8.2f}')
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()
        model.load_state_dict(torch.load(best_model_params_path))


vocab, train_data, val_data, test_data = make_data()
model = TransformerModel(len(vocab)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
lr = 5.0  # initial learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
train(model, criterion, train_data, val_data, scheduler, optimizer)

test_loss = evaluate(model, criterion, test_data)
test_ppl = math.exp(test_loss)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test perplexity {test_ppl:8.2f}')
print('=' * 89)
