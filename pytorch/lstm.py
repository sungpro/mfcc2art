# Sung Hah Hwang
# sunghahh@gmail.com


from __future__ import print_function
from dataset import Dataset
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable


# Params
sequence_length = 13
input_size = 3
output_size = 16
hidden_size = 100
num_layers = 1
num_epochs = 100
batch_size = 128
learning_rate = 0.003

torch.manual_seed(777) # for reproducibility

# Input pipeline
train_dataset = Dataset('MFCC_C.archive', 'ART_C.archive')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_dataset = Dataset('MFCC_C_test.archive', 'ART_C_test.archive')
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        # set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())

        out, _ = self.lstm(x, (h0, c0))

        # decode the hidden state of the last timestep
        out = self.fc(out[:, -1, :])
        return out


# Define net
net = Net(input_size, hidden_size, num_layers, output_size)
net.cuda()

#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def train(num_epochs):
    net.train()

    for epoch in range(num_epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            MFCC, ART = batch['MFCC'], batch['ART']
            MFCC = Variable(MFCC.view(-1, sequence_length, input_size)).cuda()
            ART = Variable(ART).cuda()
            pred = net(MFCC)
            loss = criterion(pred, ART)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{:6d}/{:6d}],  Step [{:10d}/{:10d}],  Loss: {:8.4f}'
                    .format(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))


def test():
    net.eval()

    test_loss = 0
    for i, batch in enumerate(test_loader):
        MFCC, ART = batch['MFCC'], batch['ART']
        MFCC = Variable(MFCC.view(-1, sequence_length, input_size)).cuda()
        ART = Variable(ART).cuda()
        pred = net(MFCC)
        test_loss += criterion(pred, ART, size_average=False).data[0]

    test_loss /= batch_size
    print('Step [{:10d}/{:10d}],  Loss: {:8.4f}'
                .format(i+1, len(test_dataset)//batch_size, test_loss.data[0]))


train(num_epochs)
test()



# Save
torch.save(net.state_dict(), 'mfcc2art_rnn_model.pkl')