# Sung Hah Hwang
# sunghahh@gmail.com


from __future__ import print_function
import time
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from dataset import Dataset


# Params
nseq = 13
ninp = 3
nout = 16
nhid = 100
nlayers = 1
bsz = 128
num_epochs = 100
log_interval = 200
lr = 10
clip_max_norm = 0.25
criterion = nn.MSELoss()


torch.manual_seed(777) # for reproducibility

# Input pipeline
train_dataset = Dataset('MFCC_C.archive', 'ART_C.archive')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=bsz,
                                           shuffle=True)

eval_dataset = Dataset('MFCC_C_eval.archive', 'ART_C_eval.archive')
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                          batch_size=bsz,
                                          shuffle=True)

test_dataset = Dataset('MFCC_C_test.archive', 'ART_C_test.archive')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=bsz,
                                          shuffle=True)


class Net(nn.Module):

    def __init__(self, rnn_type, nout, ninp, nhid, nlayers, dropout=0.5):

        super().__init__()

        if rnn_type in ['LSTM', 'GRU']:
            pass
        else:
            raise ValueError('Invalid option supplied')

        self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(nhid, nout)
        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers


    def init_weights(self):
        initrange = 0.1
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.fill_(0)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_().cuda()))
        else:
            return (weight.new(self.nlayers, bsz, self.nhid).zero_()).cuda()


    def forward(self, input, hidden):

        out, hidden = self.rnn(input, hidden)
        # hidden state of the last timestep
        out = self.fc(out[:, -1, :])
        return out, hidden


def repackage_hidden(h):
    """
    wraps hidden states in new Variables, to detach them from their history
    """
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


# Define net
net = Net('LSTM', nout, ninp, nhid, nlayers)
net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=lr)


def train():
    net.train()
    total_loss = 0
    start_time = time.time()
    hidden = net.init_hidden(bsz)

    for i, batch in enumerate(train_loader):
        MFCC, ART = batch['MFCC'], batch['ART']
        MFCC = Variable(MFCC.view(-1, nseq, ninp)).cuda()
        ART = Variable(ART).cuda()

        # starting each batch, we detach the hidden state from how it was previously produced
        # if we didn't, the model would try backpropagating all the way to start of the dataset
        hidden = repackage_hidden(hidden)
        pred, hidden = net(MFCC, hidden)

        optimizer.zero_grad()
        loss = criterion(pred, ART)
        total_loss += loss.data
        loss.backward()
        optimizer.step()

        # clip grad norms to prevent the exploding gradient problem in RNNs
        torch.nn.utils.clip_grad_norm(net.parameters(), clip_max_norm)
        for p in net.parameters():
            p.data.add_(-lr, p.grad.data)

        if (i+1) % log_interval == 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| Epoch {:3d}/{:6d} | Batch {:5d}/{:5d} | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f}'
                  .format(epoch+1, num_epochs, i+1, len(train_dataset) // bsz, lr,
                          elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()


def evaluate(data_loader):
    net.eval()
    total_loss = 0
    hidden = net.init_hidden(bsz)
    for i, batch in enumerate(data_loader):
        MFCC, ART = batch['MFCC'], batch['ART']
        MFCC = Variable(MFCC.view(-1, nseq, ninp), volatile=True).cuda()
        ART = Variable(ART).cuda()
        pred, hidden = net(MFCC, hidden)
        total_loss += len(MFCC) * criterion(pred, ART).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)



best_val_loss = None

try:
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(eval_loader)
        print('-' * 80)
        print('| End of epoch {:3d} | time: {:5.2f}s | valid loss: {:5.2f}'
              .format(epoch+1, (time.time() - epoch_start_time), val_loss))
        print('-' * 80)
        # save the model if the validation loss is the best we've seen so far
        if not best_val_loss or val_loss < best_val_loss:
            with open('model.pt', 'wb') as f:
                torch.save(net, f)
            best_val_loss = val_loss
        else:
            # anneal the learning rate if no improvement has been seen in the validation dataset
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 80)
    print('Exiting from training early')

# load the best saved model
with open(args.save, 'rb') as f:
    net = torch.load(f)

# run on test data
test_loss = evaluate(test_loader)
print('=' * 80)
print('| Training complete | test loss {:5.2f}'.format(test_loss))
print('=' * 80)
