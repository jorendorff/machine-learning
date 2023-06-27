import math, time
import torch, torchtext
from torch import nn
from tutorial_model import TransformerModel


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iter = torchtext.datasets.WikiText2(split='train')
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
vocab = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])


def batchify(lines, k):
    # tokenize `lines` and concat into one huge flat Tensor
    data = [torch.tensor(vocab(tokenizer(line)), dtype=torch.long) for line in lines]
    data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    # split the data into `k` columns, I guess for SIMD-style parallelism
    num_rows = data.size(0) // k
    data = data[:num_rows * k]
    data = data.view(k, num_rows).t().contiguous()
    return data.to(DEVICE) # shape: [len(data)//k, k]


train_iter, val_iter, test_iter = torchtext.datasets.WikiText2()
train_data = batchify(train_iter, 20)  # shape `[train_ntokens//20, 20]`
val_data = batchify(val_iter, 10)
test_data = batchify(test_iter, 10)


bptt = 35  # preferred training batch size divided by `k`
def get_batch(source, i):
    n = min(bptt, len(source) - 1 - i)
    data = source[i:i+n]
    target = source[i+1:i+1+n]
    return data, target


def train_one_epoch(epoch: int):
    model.train()  # turn on train mode

    start_time = time.time()
    avg_loss = 0.0
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        output = model(data)
        loss = criterion(output.view(-1, model.ntokens), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        avg_loss = 0.95 * avg_loss + 0.05 * loss.item()
        print('\r  epoch {:3d} | batch {:5d} | {:5.2f} ms/batch | loss {:5.2f} | perplexity {:5.2f}  '.format(epoch + 1, batch + 1, (time.time() - start_time) * 1000 / (batch + 1), avg_loss, math.exp(avg_loss)), end='')
    print()


def evaluate(eval_data):
    model.eval()  # turn on evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            output = model(data)
            loss = criterion(output.view(-1, model.ntokens), targets.reshape(-1))
            total_loss += data.size(0) * loss.item()
    return total_loss / len(eval_data)


model = TransformerModel(len(vocab)).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=5.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

for epoch in range(3):
    epoch_start_time = time.time()
    train_one_epoch(epoch)
    val_loss = evaluate(val_data)
    elapsed = time.time() - epoch_start_time
    print('end of epoch {:3d} | time: {:5.2f}s | eval loss {:5.2f} | perplexity {:5.2f}'.format(epoch + 1, elapsed, val_loss, math.exp(val_loss)))
    scheduler.step()

test_loss = evaluate(test_data)
print(f'end of training | test loss {test_loss:5.2f}')
torch.save(model.state_dict(), "./language-model.pt")
